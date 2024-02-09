import math
import copy
from contextlib import contextmanager
from functools import partial, wraps

import torch
import torch.nn.functional as F
import torch.distributed as distributed
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from x_clip.mlm import MLM
from x_clip.visual_ssl import SimSiam, SimCLR
from x_clip.distributed import all_gather

# helper functions

def identity(t, *args, **kwargs):
    return t

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def matrix_diag(t):
    device = t.device
    if t.ndim==3: #(mxn,b,b)
        i, j = t.shape[-2:]
    elif t.ndim==4: #(mxn,b,t,t)
        b, i, j = t.shape[-3:]
    else:
        raise NotImplementedError('input dimension should be 3 or 4')
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    if t.ndim==3:
        return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)
    elif t.ndim==4:
        return rearrange(diag_el, '(m b d) -> m b d', b=b, d=num_diag_el)

# checkpointing helper function

def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def min_max_norm(sim, axis=-1): # axis should be direction of comparison
    min_val = torch.min(sim, dim=axis, keepdim=True)[0]
    max_val = torch.max(sim, dim=axis, keepdim=True)[0]
    norm_sim = (sim-min_val)/(max_val-min_val)
    return norm_sim

def devide_by_sum(sim, axis=-1):
    sum_val = torch.sum(sim, dim=axis, keepdim=True)
    norm_sim = sim/sum_val
    return norm_sim


# helper classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))


# contrastive learning functions

def model_forward_with_context(
    *,
    fn,
    args,
    freeze,
):
    encoding_context = null_context if not freeze else torch.no_grad

    fn = fn.to('cuda')
    with encoding_context():
        enc = fn(*args)

        if freeze:
            enc.detach_()

    return enc

# main clip class

class SPARC(nn.Module):
    def __init__(
        self,
        *,
        image_encoder = None,
        text_encoder = None,
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 10000,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        text_dim_head = 64,
        text_has_cls_token = True,
        text_pad_id = 0,
        text_rotary_pos_emb = False,
        text_causal_mask = False,
        text_eos_id = None,
        text_encode_without_mask = False,
        visual_enc_depth = 6,
        visual_heads = 8,
        visual_dim_head = 64,
        visual_image_size = 256,
        visual_patch_size = 32,
        visual_patch_dropout = 0.5,
        visual_has_cls_token = True,
        channels = 3,
        use_all_token_embeds = True,
        downsample_image_embeds = False,
        decoupled_contrastive_learning = False,
        extra_latent_projection = False,
        use_mlm = False,
        text_ssl_loss_weight = 0.05,
        use_visual_ssl = False,
        visual_ssl = None,
        visual_ssl_type = 'simsiam',
        visual_ssl_hidden_layer = -1,
        simclr_temperature = 0.1,
        image_ssl_loss_weight = 0.05,
        multiview_loss_weight = 0.1,
        checkpoint_during_training = False,
        sim_reg_loss_weight = 0.,
        **kwargs
    ):
        super().__init__()
        assert use_all_token_embeds or (visual_has_cls_token or text_has_cls_token), 'CLS token must be included on both vision and text transformers if you are not using fine-grained contrastive learning loss'

        # store some parameters for access

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        # instantiate text transformer

        self.text_pad_id = text_pad_id
        self.text_has_cls_token = text_has_cls_token
        self.text_seq_len = text_seq_len

        self.text_encode_without_mask = text_encode_without_mask # whether to pass in text mask to text encoder

        self.text_causal_mask = text_causal_mask
        self.text_eos_id = text_eos_id

        assert not (text_causal_mask and not exists(text_eos_id)), 'text EOS token id must be given if using causal mask in text transformer'

        # if exists(text_encoder):
        #     self.text_transformer = text_encoder
        # else:
        #     self.text_transformer = TextTransformer(
        #         dim = dim_text,
        #         num_tokens = num_text_tokens + (1 if use_mlm else 0),
        #         max_seq_len = text_seq_len,
        #         depth = text_enc_depth,
        #         heads = text_heads,
        #         causal = text_causal_mask,
        #         dim_head = text_dim_head,
        #         rotary_pos_emb = text_rotary_pos_emb,
        #         checkpoint_during_training = checkpoint_during_training
        #     )
        assert isinstance(text_encoder, nn.Module), "'text_encoder' is required"
        self.text_transformer = text_encoder

        # instantiate image transformer

        self.visual_has_cls_token = visual_has_cls_token
        # if exists(image_encoder):
        #     self.visual_transformer = image_encoder
        # else:
        #     self.visual_transformer = VisionTransformer(
        #         dim = dim_image,
        #         image_size = visual_image_size,
        #         patch_size = visual_patch_size,
        #         channels = channels,
        #         depth = visual_enc_depth,
        #         heads = visual_heads,
        #         dim_head = visual_dim_head,
        #         patch_dropout = visual_patch_dropout,
        #         checkpoint_during_training = checkpoint_during_training
        #     )
        assert isinstance(image_encoder, nn.Module), "'image_encoder' is required"
        self.visual_transformer = image_encoder

        # text ssl

        self.use_mlm = use_mlm
        self.text_ssl_loss_weight = text_ssl_loss_weight if use_mlm else 0

        if use_mlm:
            mlm_kwargs, kwargs = groupby_prefix_and_trim('mlm_', kwargs)

            self.mlm = MLM(
                self.text_transformer,
                dim = dim_text,
                num_tokens = num_text_tokens,
                **mlm_kwargs
            )

        # image ssl

        self.use_visual_ssl = use_visual_ssl or exists(visual_ssl)
        self.image_ssl_loss_weight = image_ssl_loss_weight if use_visual_ssl else 0

        if self.use_visual_ssl:
            if exists(visual_ssl):
                self.visual_ssl = visual_ssl

            elif use_visual_ssl:
                if visual_ssl_type == 'simsiam':
                    ssl_type = partial(SimSiam, channels = channels)
                elif visual_ssl_type == 'simclr':
                    ssl_type = partial(SimCLR, temperature = simclr_temperature, channels = channels)
                else:
                    raise ValueError(f'unknown visual_ssl_type')

                self.visual_ssl = ssl_type(
                    self.visual_transformer,
                    image_size = visual_image_size,
                    hidden_layer = visual_ssl_hidden_layer
                )

        # text latent projection

        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        # image latent projection

        if downsample_image_embeds:
            assert use_all_token_embeds, 'must be using all token embeds for contrastive learning in order to downsampling'

            self.to_visual_latent = nn.Sequential(
                RearrangeImage(),
                nn.Conv2d(dim_image, dim_image, 4, stride = 2, padding = 1, bias = False, groups = dim_image),
                nn.Conv2d(dim_image, dim_latent, 1),
                Rearrange('b c h w -> b (h w) c')
            )
        else:
            self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        # temperature

        self.temperature = nn.Parameter(torch.tensor(1.))

        # from https://arxiv.org/abs/2111.07783 (FILIP paper)
        self.use_all_token_embeds = use_all_token_embeds

        # proposed in https://arxiv.org/abs/2110.06848 (DCL) and https://arxiv.org/abs/2110.11316 (CLOOB)
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        # proposed in https://arxiv.org/abs/2110.11316 (CLOOB)
        self.extra_latent_projection = extra_latent_projection

        self.to_text_latent_extra = copy.deepcopy(self.to_text_latent)
        self.to_visual_latent_extra = copy.deepcopy(self.to_visual_latent)

        self.multiview_loss_weight = multiview_loss_weight

        # is distributed or not
        self.requires_all_gather = distributed.is_initialized() and distributed.get_world_size() > 1

        # use the similarity regularization proposed in https://arxiv.org/abs/2309.08773
        self.sim_reg_loss_weight = sim_reg_loss_weight
        self.has_sim_reg_loss = sim_reg_loss_weight > 0.

    def forward(
        self,
        text,
        image,
        return_loss = False,
        return_encodings = False,
        return_latents = False,
        freeze_image_encoder = False,   # image encoder is not trained if this is set to True, proposed by LiT paper
        freeze_text_encoder = False,    # text encoder is not trained if this is set to True
        text_to_image = True,           # in the case the extra projection is turned on, would return different similarity values depending on modality directionality
        aug_text = None,                # augmented text (for multiview)
        aug_image = None                # augmented image (for multiview)
    ):
        batch, device = text.shape[0], text.device

        # derive text mask

        text_mask = text != self.text_pad_id

        # ssl

        text_ssl_loss = 0
        image_ssl_loss = 0

        if return_loss:
            text_ssl_loss = self.mlm(text, mask = text_mask) if self.use_mlm else 0
            image_ssl_loss = self.visual_ssl(image) if self.use_visual_ssl else 0

        # concat augmented texts and images and do some asserts

        num_batch_texts = num_batch_images = 1

        if exists(aug_text):
            aug_text = cast_tuple(aug_text)
            assert all(map(lambda t: t.shape == text.shape, aug_text))
            num_batch_texts = len(aug_text) + 1

            aug_text = torch.cat(aug_text, dim = 0)

            aug_text_mask = aug_text != self.text_pad_id

            text_mask = torch.cat((text_mask, aug_text_mask), dim = 0)
            text = torch.cat((text, aug_text), dim = 0)

        if exists(aug_image):
            aug_image = cast_tuple(aug_image)
            assert all(map(lambda i: i.shape == image.shape, aug_image))
            num_batch_images = len(aug_image) + 1

            aug_image = torch.cat(aug_image, dim = 0)

            image = torch.cat((image, aug_image), dim = 0)

        is_multiview = (num_batch_texts > 1 or num_batch_images > 1)
        assert not (return_loss and not self.training), 'loss cannot be used if not training'
        assert not (not return_loss and is_multiview), 'do not pass in augmented texts or images if not training'
        assert not (self.multiview_loss_weight == 0 and is_multiview), 'multiview loss weight cannot be 0 if augmented text or images passed in'

        # get encoded text

        text_args = (text,)

        if not self.text_encode_without_mask:
            text_args = (*text_args, text_mask)

        enc_text = model_forward_with_context(
            fn = self.text_transformer,
            args = text_args,
            freeze = freeze_text_encoder
        )

        # depending on whether text is using causal mask, post process, moving eos token to the first position

        if self.text_causal_mask:
            eos_text_mask = (text == self.text_eos_id)
            assert torch.all(torch.any(eos_text_mask, dim = -1)), f'some of the text rows does not have the eos id {self.text_eos_id}'

            text_len = text.shape[-1]
            eos_indices = eos_text_mask.float().argmax(dim = -1, keepdim = True)

            eos_text_mask = torch.zeros_like(eos_text_mask).scatter(1, eos_indices, 1.).bool()
            eos_text_mask = rearrange(eos_text_mask, '... -> ... 1')

            eos_tokens = enc_text.masked_select(eos_text_mask)
            rest_tokens = enc_text.masked_select(~eos_text_mask)

            eos_tokens = rearrange(eos_tokens, '(b d) -> b 1 d', b = b)
            rest_tokens = rearrange(rest_tokens, '(b n d) -> b n d', b = b, n = text_len - 1)
            enc_text = torch.cat((eos_tokens, rest_tokens), dim = 1)

        # whether to train image encoder, in the case that the image net was pretrained as recommended in LiT
        enc_image = model_forward_with_context(
            fn = self.visual_transformer,
            args = (image,),
            freeze = freeze_image_encoder
        )

        # early return of encodings, if needed (for DALL-E2)

        if return_encodings:
            return enc_text, enc_image

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only
        
        if self.use_all_token_embeds: # get both local, global latent
            assert enc_text.ndim == 3, 'encoded text must have 3 dimensions (batch, seq, features)'
            assert enc_image.ndim == 3, 'encoded image must have 3 dimensions (batch, seq [height x width], features)'
            local_text_embeds = enc_text[:, 1:] if self.text_has_cls_token else enc_text
            local_image_embeds = enc_image[:, 1:] if self.visual_has_cls_token else enc_image

            glob_text_embeds = enc_text[:, 0] if enc_text.ndim == 3 else enc_text
            glob_image_embeds = enc_image[:, 0] if enc_image.ndim == 3 else enc_image

        # project to latents
        glob_text_latents = self.to_text_latent(glob_text_embeds) # TODO: need non-linear layer before converting to latent
        glob_image_latents = self.to_visual_latent(glob_image_embeds) # TODO: need non-linear layer before converting to latent
        glob_text_latents, glob_image_latents = map(l2norm, (glob_text_latents, glob_image_latents))

        local_text_latents = self.to_text_latent(local_text_embeds)
        local_image_latents = self.to_visual_latent(local_image_embeds)

        # calculate another set of latents for image to text (vs text to image)
        # proposed by CLOOB

        text_latents_extra, image_latents_extra = glob_text_latents, glob_image_latents
        if self.extra_latent_projection:
            text_latents_extra = self.to_text_latent_extra(glob_text_embeds)
            image_latents_extra = self.to_visual_latent_extra(glob_image_embeds)
            text_latents_extra, image_latents_extra = map(l2norm, (text_latents_extra, image_latents_extra))

        # whether to early return latents

        if return_latents:
            if self.extra_latent_projection:
                return glob_text_latents, glob_image_latents, text_latents_extra, image_latents_extra

            return glob_text_latents, glob_image_latents

        # get temperature

        temp = self.temperature.exp()

        # early return, if needed

        if not return_loss and self.use_all_token_embeds:
            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (local_text_latents, local_image_latents)
            return einsum('b t d, b i d -> b t i', *einsum_args) * temp

        if not return_loss and not self.use_all_token_embeds:
            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (glob_text_latents, glob_image_latents)
            return einsum('b d, b d -> b', *einsum_args) * temp

        # split out multiview dimension for text and images

        glob_text_latents = rearrange(glob_text_latents, '(m b) ... -> m b ...', m = num_batch_texts) # (m b d)
        glob_image_latents = rearrange(glob_image_latents, '(m b) ... -> m b ...', m = num_batch_images)
        local_text_latents = rearrange(local_text_latents, '(m b) ... -> m b ...', m = num_batch_texts) # (m b t d)
        local_image_latents = rearrange(local_image_latents, '(m b) ... -> m b ...', m = num_batch_texts)

        if self.extra_latent_projection:
            text_latents_extra = rearrange(text_latents_extra, '(m b) ... -> m b ...', m = num_batch_texts)
            image_latents_extra = rearrange(image_latents_extra, '(m b) ... -> m b ...', m = num_batch_images)

        # maybe distributed all gather

        if self.requires_all_gather:
            latents = torch.stack((glob_text_latents, glob_image_latents))
            latents, sizes = all_gather(latents, 2, None)
            glob_text_latents, glob_image_latents = latents

            batch = sizes.sum().item()

            if self.extra_latent_projection:
                latents_extra = torch.stack((text_latents_extra, image_latents_extra))
                latents_extra, _ = all_gather(latents_extra, 2, sizes)
                text_latents_extra, image_latents_extra = latents_extra

        # maybe similarity regularize

        sim_reg_loss = 0.

        if self.has_sim_reg_loss: 
            diag_mask = torch.eye(batch, device = device, dtype = torch.bool)
            off_diag_mask = rearrange(~diag_mask, '... -> 1 ...')

            text_sim, image_sim, text_extra_sim, image_extra_sim = map(lambda t: einsum('m i ... d, m j ... d -> m ... i j', t, t)[off_diag_mask], (glob_text_latents, glob_image_latents, text_latents_extra, image_latents_extra))

            sim_reg_loss = (
                F.mse_loss(text_sim, image_sim) +
                F.mse_loss(text_extra_sim, image_extra_sim)
            ) / 2

        # contrastive loss

        """
        m - num batches of text (for multiview)
        n - num batches of images (for multiview)
        x - batches of text
        y - batches of images
        t - sequence dimension along text tokens
        i - sequence dimension along image tokens
        """
        # import ipdb; ipdb.set_trace()
        if self.use_all_token_embeds:
            # --- global clip loss ---
            global_text_to_image = einsum('m x d, n y d -> m n x y', glob_text_latents, glob_image_latents) * temp # (m,b,d), (n,b,d) -> (m,n,b,b)
            global_image_to_text = rearrange(global_text_to_image, '... t i -> ... i t')

            if self.extra_latent_projection:
                image_to_text = einsum('m t d, n i d -> m n i t', text_latents_extra, image_latents_extra) * temp

            # --- fine-grained CLIP logic ---
            sim_text_to_image = einsum('m x t d, n y i d -> m n x y t i', local_text_latents, local_image_latents) * temp
            # sparsify & normalize
            sparse_text_to_image = min_max_norm(sim_text_to_image, axis=-1)
            threshold = 1/sim_text_to_image.size(-1) # threshold=1/P
            sparse_text_to_image = torch.where(sparse_text_to_image<threshold, torch.zeros_like(sparse_text_to_image), sparse_text_to_image) # convert to 0 when sim lower than threshold
            sparse_text_to_image = devide_by_sum(sparse_text_to_image, axis=-1) 
            # alignment-weighting
            text_grouped_image_latents = einsum('m n x y t i, n y i d -> m n x y t d', sparse_text_to_image, local_image_latents)
            text_grouped_image_latents, local_text_latents = map(l2norm, (text_grouped_image_latents, local_text_latents))
            # only select positive pair
            text_grouped_image_latents_pos = [text_grouped_image_latents[:,:,x,x,:,:] for x in range(text_grouped_image_latents.size(2))]
            text_grouped_image_latents = torch.stack(text_grouped_image_latents_pos, dim=2) # (m,n,x,t,d)
            # calculate sim
            local_text_to_grouped = einsum('m x t d, m n x i d -> m n x t i', local_text_latents, text_grouped_image_latents) * temp
            # mask
            text_to_grouped_mask = rearrange(text_mask, '(m b) t -> m 1 b t 1', m=num_batch_texts)
            grouped_to_text_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t', m=num_batch_texts)
            overall_mask = einsum('m n b t i, m n b i k -> m n b t k', text_to_grouped_mask, grouped_to_text_mask) # (m,n,x,t,t)
            local_text_to_grouped = local_text_to_grouped.masked_fill(~overall_mask, 0.) * torch.ones(overall_mask.shape).sum() / overall_mask.sum()
            local_grouped_to_text = rearrange(local_text_to_grouped, '... t i -> ... i t')


        # calculate loss

        global_text_to_image = rearrange(global_text_to_image, 'm n ... -> (m n) ...') #(m,n,b,b) -> (mxn,b,b)
        global_image_to_text = rearrange(global_image_to_text, 'm n ... -> (m n) ...') 
        local_text_to_grouped = rearrange(local_text_to_grouped, 'm x ... -> (m x) ...') #(m,n,b,t,t) -> (mxn,b,t,t)
        local_grouped_to_text = rearrange(local_grouped_to_text, 'm x ... -> (m x) ...')

        # exponentiate

        text_to_image_exp, image_to_text_exp = map(torch.exp, (global_text_to_image, global_image_to_text))
        text_to_grouped_exp, grouped_to_text_exp = map(torch.exp, (local_text_to_grouped, local_grouped_to_text))
        
        # numerators
        text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp)) #(mxn,b)
        text_to_grouped_pos, grouped_to_text_pos = map(matrix_diag, (text_to_grouped_exp, grouped_to_text_exp)) #(mxn,b,t)

        # denominator

        if self.decoupled_contrastive_learning:
            pos_mask = torch.eye(batch, device = device, dtype = torch.bool)
            text_to_image_exp, image_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (text_to_image_exp, image_to_text_exp))
            # [TODO]: grouped not implemented
            raise NotImplementedError('local grouped loss not implemented on decoupled_contrastive_learning')

        text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_image_exp, image_to_text_exp)) #(mxn,b)
        text_to_grouped_denom, grouped_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_grouped_exp, grouped_to_text_exp)) #(mxn,b,t)

        # loss

        text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(dim = -1) #(mxn)
        image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(dim = -1)
        text_to_grouped_loss = (-log(text_to_grouped_pos) + log(text_to_grouped_denom)).mean(dim = -1).mean(dim = -1) #(mxn)
        grouped_to_text_loss = (-log(grouped_to_text_pos) + log(grouped_to_text_denom)).mean(dim = -1).mean(dim = -1)


        # calculate CL loss

        global_cl_losses = (text_to_image_loss + image_to_text_loss) / 2
        local_cl_losses = (text_to_grouped_loss + grouped_to_text_loss) / 2


        # get main CL loss vs multiview CL losses

        global_cl_loss, global_multiview_cl_loss = global_cl_losses[0], global_cl_losses[1:]
        local_cl_loss, local_multiview_cl_loss = local_cl_losses[0], local_cl_losses[1:]


        # if no augmented text or images passed in, multiview loss weight is 0

        multiview_loss_weight = self.multiview_loss_weight if is_multiview else 0

        # calculate weights

        cl_loss_weight = 1 - (self.text_ssl_loss_weight + self.image_ssl_loss_weight + multiview_loss_weight)

        loss = (global_cl_loss * cl_loss_weight/2) \
            + (local_cl_loss * cl_loss_weight/2) \
            + (text_ssl_loss * self.text_ssl_loss_weight) \
            + (image_ssl_loss * self.image_ssl_loss_weight)

        # add multiview CL loss with weight

        if is_multiview:
            loss = loss + (global_multiview_cl_loss.mean() + local_multiview_cl_loss.mean()) * multiview_loss_weight/2

        # add similarity regularization loss with weight if needed

        if self.has_sim_reg_loss:
            loss = loss + sim_reg_loss * self.sim_reg_loss_weight

        return loss

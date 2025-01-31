import torch

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs[
            "include_input"
        ]:  # original raw input "x" is also included in the output
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d  # each has a dim of d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, scalar_factor=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder = Embedder(**embed_kwargs)
    embed = lambda x, embedder=embedder: embedder.embed(x / scalar_factor)
    return embed, embedder.out_dim


def fc_block(in_f, out_f):
    return nn.Sequential(nn.Linear(in_f, out_f), nn.ReLU())


class Semantic_NeRF(nn.Module):
    """
    Compared to the NeRF class which also predicts semantic logits from MLPs,
    here we make the semantic label only a function of 3D position
    instead of both position and viewing directions.
    """

    def __init__(
        self,
        enable_multitask,
        enable_semantic,
        enable_surface_normal,
        num_semantic_classes,
        d=8,
        w=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=None,
        use_viewdirs=False,
    ):
        super(Semantic_NeRF, self).__init__()
        if skips is None:
            skips = [4]
        """
                D: number of layers for density (sigma) encoder
                W: number of hidden units in each layer
                input_ch: number of input channels for xyz (3+3*10*2=63 by default)
                in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
                skips: layer index to add skip connection in the Dth layer
        """
        self.d = d
        self.w = w
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.enable_multitask = enable_multitask
        self.enable_semantic = enable_semantic
        self.enable_surface_normal = enable_surface_normal

        # build the encoder
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, w)]
            + [
                nn.Linear(w, w) if i not in self.skips else nn.Linear(w + input_ch, w)
                for i in range(d - 1)
            ]
        )

        # Implementation according to the official code release below
        # https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105

        # Another layer is used to
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + w, w // 2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(w, w)
            self.alpha_linear = nn.Linear(w, 1)
            self.rgb_linear = nn.Linear(w // 2, 3)
            if enable_multitask:  # here to output everything apart from rgb
                self.semantic_linear = nn.Sequential(
                    fc_block(w, w // 2), nn.Linear(w // 2, num_semantic_classes)
                )
                self.surface_normal_linear = nn.Linear(w // 2, 3)
        else:
            self.output_linear = nn.Linear(w, output_ch)

    def forward(self, x, show_endpoint=False):
        """
        Encodes input (xyz+dir) to rgb+sigma+semantics raw output
        Inputs:
            x: [B, self.in_channels_xyz(+self.in_channels_dir)]
                the embedded vector of 3D xyz position and viewing direction
        """
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # if using view-dirs, output occupancy alpha as well as features for concatenation
            alpha = self.alpha_linear(h)  # [..., 1]
            if self.enable_multitask:
                if self.enable_semantic:
                    sem_logits = self.semantic_linear(h)  # [..., num_classes]
                else:
                    sem_logits = torch.zeros(list(alpha.shape[:-1]) + [28], requires_grad=False).cuda()

            # prepare view dependent feature
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            if show_endpoint:
                endpoint_feat = h
            rgb = self.rgb_linear(h)  # [..., 3]
            if self.enable_multitask:
                if self.enable_surface_normal: # surface normal is view dependant
                    surface_normal = self.surface_normal_linear(h)
                else:
                    surface_normal = torch.zeros(list(alpha.shape[:-1]) + [3], requires_grad=False).cuda()
                    
            if self.enable_multitask:
                outputs = torch.cat([rgb, alpha, surface_normal, sem_logits], -1)
            else:
                outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        if show_endpoint is False:
            return outputs
        else:
            return torch.cat([outputs, endpoint_feat], -1)

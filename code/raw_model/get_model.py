"""Model factory. Legacy MAMCA dependencies are imported only on request."""


def get_model(args, device="cuda:0"):
    name = args.arch
    num_classes = len(args.class_names)
    if name in ("MAMC2", "MAMC2-CSP"):
        from mamc import MAMC2, MAMC2Config
        options = {key: getattr(args, key) for key in MAMC2Config.__dataclass_fields__
                   if hasattr(args, key)}
        options["num_classes"] = num_classes
        return MAMC2(MAMC2Config(**options)).to(device)
    if name == "MAMCA":
        from .MAMCA.MAMCA import MAMCA
        from mamba_ssm.models.config_mamba import MambaConfig
        config = MambaConfig(d_model=16, n_layer=1,
                             ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
                             rms_norm=True, fused_add_norm=False)
        return MAMCA(config=config, device=device, length=int(args.length_train),
                     num_claasses=num_classes).to(device)
    raise NotImplementedError(f"Model {name} is not implemented")

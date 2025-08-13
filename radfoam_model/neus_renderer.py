import torch


class ErrorBox:
    def __init__(self):
        self.ray_error = None
        self.point_error = None


class TraceRaysNeuS(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        neus_pipeline,
        sdf_field,
        color_field,
        deviation_field,
        rays,
        start_point,
        depth_quantiles,
        return_contribution,
    ):
        ctx.rays = rays
        ctx.start_point = start_point
        ctx.depth_quantiles = depth_quantiles
        ctx.neus_pipeline = neus_pipeline
        ctx.sdf_field = sdf_field
        ctx.color_field = color_field
        ctx.deviation_field = deviation_field

        results = neus_pipeline.trace_forward(
            sdf_field,
            color_field,
            deviation_field,
            rays,
            start_point,
            depth_quantiles=depth_quantiles,
            return_contribution=return_contribution,
            mode=2,
        )
        ctx.rgba = results["rgba"]
        ctx.depth_indices = results.get("depth_indices", None)

        errbox = ErrorBox()
        ctx.errbox = errbox

        return (
            results["rgba"],
            results.get("depth", None),
            results.get("contribution", None),
            results["num_intersections"],
            errbox,
        )

    @staticmethod
    def backward(
        ctx,
        grad_rgba,
        grad_depth,
        grad_contribution,
        grad_num_intersections,
        errbox_grad,
    ):
        del grad_contribution
        del grad_num_intersections
        del errbox_grad

        rays = ctx.rays
        start_point = ctx.start_point
        neus_pipeline = ctx.neus_pipeline
        rgba = ctx.rgba
        sdf_field = ctx.sdf_field
        color_field = ctx.color_field
        deviation_field = ctx.deviation_field
        depth_quantiles = ctx.depth_quantiles

        results = neus_pipeline.trace_backward(
            sdf_field,
            color_field,
            deviation_field,
            rays,
            start_point,
            rgba,
            grad_rgba,
            depth_quantiles,
            ctx.depth_indices,
            grad_depth,
            ctx.errbox.ray_error,
            mode=2,
        )
        sdf_grad = results["sdf_grad"]
        color_grad = results["color_grad"]
        deviation_grad = results["deviation_grad"]
        ctx.errbox.point_error = results.get("point_error", None)

        # Handle NaN gradients
        if sdf_grad is not None:
            sdf_grad[~sdf_grad.isfinite()] = 0
        if color_grad is not None:
            color_grad[~color_grad.isfinite()] = 0
        if deviation_grad is not None:
            deviation_grad[~deviation_grad.isfinite()] = 0

        del (
            ctx.rays,
            ctx.start_point,
            ctx.neus_pipeline,
            ctx.rgba,
            ctx.sdf_field,
            ctx.color_field,
            ctx.deviation_field,
            ctx.depth_quantiles,
        )
        return (
            None,  # neus_pipeline
            sdf_grad,  # sdf_field
            color_grad,  # color_field
            deviation_grad,  # deviation_field
            None,  # rays
            None,  # start_point
            None,  # depth_quantiles
            None,  # return_contribution
        )

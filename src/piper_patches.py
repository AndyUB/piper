"""
Runtime monkey patches for PyTorch and Ray dependencies.

These patches enable Piper's RemoteTensor to work correctly with TorchDynamo
and Ray's tensor transport backends.

Import this module early in your application to apply patches:
    import src.piper_patches
"""

import functools
import logging

logger = logging.getLogger(__name__)

_patches_applied = False


def apply_patches():
    """Apply all required monkey patches to PyTorch and Ray."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    _patch_torch_dynamo_compile()
    _patch_torch_dynamo_guards()
    _patch_ray_actor_method()

    logger.info("Piper runtime patches applied successfully")


def _patch_torch_dynamo_compile():
    """
    Patch torch._dynamo.convert_frame._compile to handle RemoteTensor.

    Instead of tracing RemoteTensors, trace their underlying FakeTensor.
    This prevents TorchDynamo from failing on non-traceable RemoteTensors.
    """
    try:
        import torch._dynamo.convert_frame as convert_frame
    except ImportError as e:
        logger.warning(f"Could not import convert_frame: {e}")
        return

    # Import RemoteTensor lazily to avoid circular imports
    def get_remote_tensor_class():
        try:
            from src.piper_utils import RemoteTensor
            return RemoteTensor
        except ImportError:
            return None

    original_compile = convert_frame._compile

    @functools.wraps(original_compile)
    def patched_compile(code, globals, locals, builtins, closure, *args, **kwargs):
        RemoteTensor = get_remote_tensor_class()
        if RemoteTensor is not None:
            # Replace RemoteTensors with their underlying FakeTensor in locals
            patched_locals = {}
            for k, v in locals.items():
                if isinstance(v, RemoteTensor):
                    patched_locals[k] = v._fake
                else:
                    patched_locals[k] = v
            locals = patched_locals

        return original_compile(code, globals, locals, builtins, closure, *args, **kwargs)

    convert_frame._compile = patched_compile
    logger.debug("Patched torch._dynamo.convert_frame._compile")


def _patch_torch_dynamo_guards():
    """
    Patch torch._dynamo.guards to handle RemoteTensor.

    RemoteTensor causes recompilation bugs because it's not fully traceable
    by TorchDynamo. This patch modifies guard behavior to prevent issues.
    """
    try:
        import torch._dynamo.guards as guards_module
    except ImportError as e:
        logger.warning(f"Could not import guards: {e}")
        return

    def get_remote_tensor_class():
        try:
            from src.piper_utils import RemoteTensor
            return RemoteTensor
        except ImportError:
            return None

    # Patch GuardBuilder.TENSOR_MATCH to skip guards for RemoteTensor
    if hasattr(guards_module, 'GuardBuilder'):
        original_tensor_match = guards_module.GuardBuilder.TENSOR_MATCH

        @functools.wraps(original_tensor_match)
        def patched_tensor_match(self, guard, value=None):
            RemoteTensor = get_remote_tensor_class()
            if RemoteTensor is not None:
                # Get the actual value being guarded
                actual_value = value if value is not None else self.get(guard.name)
                if isinstance(actual_value, RemoteTensor):
                    # Skip TENSOR_MATCH for RemoteTensor - it will cause recompilation issues
                    logger.debug(f"Skipping TENSOR_MATCH guard for RemoteTensor: {guard.name}")
                    return
            return original_tensor_match(self, guard, value)

        guards_module.GuardBuilder.TENSOR_MATCH = patched_tensor_match
        logger.debug("Patched torch._dynamo.guards.GuardBuilder.TENSOR_MATCH")


def _patch_ray_actor_method():
    """
    Patch ray.actor.ActorMethod._remote to support multiple return values
    with tensor transport backends.

    The default Ray implementation only supports 1 return value per task
    when using tensor transport. This patch removes that restriction and
    properly registers multiple ObjectRefs with the GPU object manager.
    """
    try:
        import ray
        import ray.actor as actor_module
        from ray._raylet import ObjectRef
    except ImportError as e:
        logger.warning(f"Could not import ray.actor: {e}")
        return

    # Get TensorTransportEnum for comparison
    try:
        from ray.actor import TensorTransportEnum
    except ImportError:
        logger.warning("Could not import TensorTransportEnum, skipping Ray patch")
        return

    original_remote = actor_module.ActorMethod._remote

    @functools.wraps(original_remote)
    def patched_remote(
        self,
        args=None,
        kwargs=None,
        name="",
        num_returns=None,
        max_task_retries=None,
        retry_exceptions=None,
        concurrency_group=None,
        _generator_backpressure_num_objects=None,
        enable_task_events=None,
        tensor_transport=None,
    ):
        if num_returns is None:
            num_returns = self._num_returns
        if max_task_retries is None:
            max_task_retries = self._max_task_retries
        if max_task_retries is None:
            max_task_retries = 0
        if retry_exceptions is None:
            retry_exceptions = self._retry_exceptions
        if enable_task_events is None:
            enable_task_events = self._enable_task_events
        if _generator_backpressure_num_objects is None:
            _generator_backpressure_num_objects = (
                self._generator_backpressure_num_objects
            )

        if tensor_transport is None:
            tensor_transport = self._tensor_transport

        # PIPER MODIFICATION: Remove the num_returns != 1 check for tensor_transport
        # Original code would raise ValueError here if num_returns != 1
        if tensor_transport != TensorTransportEnum.OBJECT_STORE.name:
            # Skip the num_returns check - allow multiple returns
            if not self._actor._ray_enable_tensor_transport:
                raise ValueError(
                    f'Currently, methods with .options(tensor_transport="{tensor_transport}") are not supported when enable_tensor_transport=False. '
                    "Please set @ray.remote(enable_tensor_transport=True) on the actor class definition."
                )
            gpu_object_manager = ray._private.worker.global_worker.gpu_object_manager
            if not gpu_object_manager.actor_has_tensor_transport(
                self._actor, tensor_transport
            ):
                raise ValueError(
                    f'{self._actor} does not have tensor transport {tensor_transport} available. If using a collective-based transport ("nccl" or "gloo"), please create a communicator with '
                    "`ray.experimental.collective.create_collective_group` "
                    "before calling actor tasks with non-default tensor_transport."
                )

        args = args or []
        kwargs = kwargs or {}

        def invocation(args, kwargs):
            dst_actor = self._actor
            if dst_actor is None:
                raise RuntimeError(
                    "Lost reference to actor. Actor handles must be stored as variables, e.g. `actor = MyActor.remote()` before calling methods."
                )

            gpu_object_manager = ray._private.worker.global_worker.gpu_object_manager
            gpu_object_manager.trigger_out_of_band_tensor_transfer(dst_actor, args)

            return dst_actor._actor_method_call(
                self._method_name,
                args=args,
                kwargs=kwargs,
                name=name,
                num_returns=num_returns,
                max_task_retries=max_task_retries,
                retry_exceptions=retry_exceptions,
                concurrency_group_name=concurrency_group,
                generator_backpressure_num_objects=(
                    _generator_backpressure_num_objects
                ),
                enable_task_events=enable_task_events,
                tensor_transport=tensor_transport,
            )

        # Apply the decorator if there is one.
        if self._decorator is not None:
            invocation = self._decorator(invocation)

        object_refs = invocation(args, kwargs)

        # PIPER MODIFICATION: Handle multiple return values with GPU object manager
        if tensor_transport != TensorTransportEnum.OBJECT_STORE.name:
            gpu_object_manager = ray._private.worker.global_worker.gpu_object_manager
            if isinstance(object_refs, ObjectRef):
                # Single return value
                object_ref = object_refs
                gpu_object_manager.add_gpu_object_ref(
                    object_ref, self._actor, tensor_transport
                )
            else:
                # Multiple return values - register each ObjectRef
                for object_ref in object_refs:
                    if isinstance(object_ref, ObjectRef):
                        gpu_object_manager.add_gpu_object_ref(
                            object_ref, self._actor, tensor_transport
                        )

        return object_refs

    actor_module.ActorMethod._remote = patched_remote
    logger.debug("Patched ray.actor.ActorMethod._remote")


apply_patches()

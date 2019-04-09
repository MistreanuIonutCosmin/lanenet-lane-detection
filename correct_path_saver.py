import tensorflow as tf


def restore_from_classification_checkpoint_fn(feature_extractor_scope):
    """Returns a map of variables to load from a foreign checkpoint.

    Args:
      feature_extractor_scope: A scope name for the feature extractor.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    variables_to_restore = {}
    for variable in tf.global_variables():
        var_name = variable.op.name
        if var_name.startswith(feature_extractor_scope + '/'):
            var_name = var_name.replace(feature_extractor_scope + '/', '')
            variables_to_restore[var_name] = variable

    return variables_to_restore


def get_variables_available_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True):
    if isinstance(variables, list):
        variable_names_map = {variable.op.name: variable for variable in variables}
    elif isinstance(variables, dict):
        variable_names_map = variables
    else:
        raise ValueError('`variables` is expected to be a list or dict.')
    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    if not include_global_step:
        ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
        ckpt_vars_to_shape_map.pop('train_op/beta1_power', None)
        ckpt_vars_to_shape_map.pop('train_op/beta2_power', None)
        ckpt_vars_to_shape_map.pop('train_op/global_step', None)
    vars_in_ckpt = {}
    for variable_name, variable in sorted(variable_names_map.items()):
        if variable_name in ckpt_vars_to_shape_map:
            if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
                vars_in_ckpt[variable_name] = variable
            else:
                tf.logging.warning('Variable [%s] is available in checkpoint, but has an '
                                   'incompatible shape with model variable. Checkpoint '
                                   'shape: [%s], model variable shape: [%s]. This '
                                   'variable will not be initialized from the checkpoint.',
                                   variable_name, ckpt_vars_to_shape_map[variable_name],
                                   variable.shape.as_list())
        else:
            tf.logging.warning('Variable [%s] is not available in checkpoint',
                               variable_name)
    if isinstance(variables, list):
        return vars_in_ckpt.values()
    return vars_in_ckpt

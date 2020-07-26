from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


def get_pretrained_model_config(pretrained_model_args, task_name, num_labels, output_dir):
    return AutoConfig.from_pretrained(
        pretrained_model_args.config_name if pretrained_model_args.config_name
        else pretrained_model_args.pretrained_model_cache_dir,
        num_labels=num_labels,
        finetuning_task=task_name,
        cache_dir=output_dir,
    )


def get_tokenizer(pretrained_model_args, output_dir):
    return AutoTokenizer.from_pretrained(
        pretrained_model_args.tokenizer_name if pretrained_model_args.tokenizer_name
        else pretrained_model_args.pretrained_model_cache_dir,
        do_lower_case=False,
        cache_dir=output_dir,
    )


def get_pretrained_model(pretrained_model_cache_dir, pretrained_model_config, output_dir):
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_cache_dir,
        from_tf=bool(".ckpt" in pretrained_model_cache_dir),
        config=pretrained_model_config,
        cache_dir=output_dir,
    )

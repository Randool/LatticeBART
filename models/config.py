from transformers import BartConfig, BartTokenizer


class LatticeBartConfig(BartConfig):
    model_type = "lattice_bart"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=50265,
            max_position_embeddings=1024,
            encoder_layers=6,
            lat_enc_layers=6,
            encoder_ffn_dim=3072,
            encoder_attention_heads=12,
            decoder_layers=6,
            decoder_ffn_dim=3072,
            decoder_attention_heads=12,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            activation_function="gelu",
            d_model=768,
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            init_std=0.02,
            classifier_dropout=0.0,
            scale_embedding=False,
            gradient_checkpointing=False,
            force_bos_token_to_be_generated=False,
            use_cache=True,
            num_labels=3,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            is_encoder_decoder=True,
            decoder_start_token_id=2,
            **kwargs
    ):
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.lat_enc_layers = lat_enc_layers
        assert lat_enc_layers <= encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.force_bos_token_to_be_generated = force_bos_token_to_be_generated  # only relevant for CNN

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model


def tiny_model_config():
    return LatticeBartConfig.from_json_file("./models/configs/tiny/config.json")


def base_model_config():
    return LatticeBartConfig.from_json_file("./models/configs/base/config.json")


def large_model_config():
    return LatticeBartConfig.from_json_file("./models/configs/large/config.json")


def get_model_config(_type: str):
    if _type == "tiny":
        return tiny_model_config()
    elif _type == "base":
        return base_model_config()
    elif _type == "large":
        return large_model_config()
    else:
        raise ValueError(f"Unrecognized type: {_type}")


def get_tokenizer():
    return BartTokenizer.from_pretrained("./models/configs/base")

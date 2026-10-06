from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from . import processing_qwen3_vl  # noqa: F401
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
):
    # Reshape the tensors to 1D
    final_embedding_shape = final_embedding.shape
    scaled_image_features_flattened = mx.flatten(scaled_image_features)
    final_embedding_flattened = mx.flatten(final_embedding)
    image_mask_expanded_flattened = mx.flatten(image_mask_expanded)

    # Scatter the scaled image features into the special image token positions
    image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
    final_embedding_flattened[image_positions] = scaled_image_features_flattened

    # Reshape back to the original shape
    final_embedding = mx.reshape(final_embedding_flattened, final_embedding_shape)

    return final_embedding


# gmlx: encode_images below gives the cache entry that this class's own
# get_input_embeddings reads. A subclass that joins the image features its
# own way (qwen3_5, qwen4_exp) reads another entry, so it does not inherit
# the method, and the generate path caches nothing for it, as before.
class _OwnJoinOnly:
    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, owner=None):
        owner = type(obj) if owner is None else owner
        if owner.get_input_embeddings is not Model.get_input_embeddings:
            raise AttributeError(self.fn.__name__)
        return self.fn.__get__(obj, owner)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = (
            None if config.skip_vision else VisionModel(config.vision_config)
        )
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        pixel_values_videos = kwargs.get("pixel_values_videos", None)

        # gmlx: cached image features stand in for the image pixels. A
        # caller passes the entry itself, or a vision cache and its key.
        vision_cache = kwargs.get("vision_cache", None)
        image_key = kwargs.get("_image_key", None)
        cached = kwargs.get("cached_image_features", None)
        if cached is None and vision_cache is not None and image_key is not None:
            cached = vision_cache.get(image_key)

        # Video inputs flow in via pixel_values_videos from the generic
        # prepare_inputs path; alias to pixel_values for the unified encoder.
        # gmlx: a call with cached features and no pixels is an image call.
        if pixel_values is None and pixel_values_videos is None and cached is None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, attention_mask=mask
            )
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids),
                position_ids=position_ids,
                rope_deltas=rope_deltas,
            )

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # gmlx: one path joins the features, cached or computed. The cached
        # branch used to join the main features and drop the deepstack
        # features, so the decoder ran without them.
        dtype = inputs_embeds.dtype
        image_features = image_deepstack = None
        video_features = video_deepstack = None
        if cached is not None:
            image_features, image_deepstack = self._split_image_features(
                cached, dtype
            )
            n_image_tokens = int((input_ids == self.config.image_token_index).sum())
            if image_features.shape[0] != n_image_tokens:
                raise ValueError(
                    "cached_image_features hold "
                    f"{image_features.shape[0]} image tokens and the prompt "
                    f"has {n_image_tokens}"
                )
        elif pixel_values is not None:
            image_features, image_deepstack = self._vision_features(
                pixel_values, image_grid_thw, dtype
            )
            if vision_cache is not None and image_key is not None:
                entry = self._pack_image_features(image_features, image_deepstack)
                mx.eval(entry)
                vision_cache.put(image_key, entry)
        if pixel_values_videos is not None:
            video_features, video_deepstack = self._vision_features(
                pixel_values_videos, video_grid_thw, dtype
            )

        image_mask = input_ids == self.config.image_token_index
        video_mask = input_ids == self.config.video_token_index
        if image_features is not None:
            inputs_embeds = masked_scatter(
                inputs_embeds,
                mx.broadcast_to(image_mask[..., None], inputs_embeds.shape),
                image_features,
            )
        if video_features is not None:
            inputs_embeds = masked_scatter(
                inputs_embeds,
                mx.broadcast_to(video_mask[..., None], inputs_embeds.shape),
                video_features,
            )
        deepstack_visual_embeds = self._merge_deepstack_features(
            input_ids,
            image_deepstack,
            video_deepstack,
        )

        visual_pos_masks = (input_ids == self.config.image_token_index) | (
            input_ids == self.config.video_token_index
        )
        if deepstack_visual_embeds is not None:
            mx.eval(deepstack_visual_embeds)

        position_ids, rope_deltas = self.language_model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, mask
        )

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
        )

    # gmlx: the tower computes in float32 whatever dtype its weights are
    # stored in. Its residual stream reaches the thousands in the last
    # blocks, and a 16-bit float drops the small values beside them: in
    # bfloat16 the main features come out about a tenth off. The features
    # take the dtype of the token embeddings here, where they join them.
    def _vision_features(self, pixel_values, grid_thw, dtype):
        if self.vision_tower is None:
            raise ValueError("this Qwen3-VL model was loaded without a vision tower")
        features, deepstack = self.vision_tower(
            pixel_values.astype(mx.float32), grid_thw
        )
        return features.astype(dtype), [d.astype(dtype) for d in deepstack]

    # gmlx: a cache entry for this class is one array, the main features
    # and then the deepstack features in block order:
    # [1 + deepstack blocks, image tokens, hidden].
    @staticmethod
    def _pack_image_features(features, deepstack):
        return mx.stack([features] + list(deepstack))

    def _split_image_features(self, cached, dtype):
        n_deepstack = len(self.config.vision_config.deepstack_visual_indexes)
        if (
            not isinstance(cached, mx.array)
            or cached.ndim != 3
            or cached.shape[0] != 1 + n_deepstack
        ):
            shape = cached.shape if isinstance(cached, mx.array) else type(cached)
            raise ValueError(
                "cached_image_features for qwen3_vl carry the main features "
                f"and the {n_deepstack} deepstack features, stacked as "
                f"[{1 + n_deepstack}, image tokens, hidden]; got {shape}. "
                "Main features alone would join without the deepstack features."
            )
        cached = cached.astype(dtype)
        return cached[0], [cached[i] for i in range(1, cached.shape[0])]

    # gmlx: the generate path fills its vision cache through this call, and
    # passes the entry back as cached_image_features.
    @_OwnJoinOnly
    def encode_images(self, pixel_values: mx.array, **kwargs) -> mx.array:
        """The cache entry of the images in ``pixel_values``."""
        dtype = self.language_model.model.embed_tokens(mx.array([[0]])).dtype
        features, deepstack = self._vision_features(
            pixel_values, kwargs.get("image_grid_thw", None), dtype
        )
        return self._pack_image_features(features, deepstack)

    def _merge_deepstack_features(
        self,
        input_ids: mx.array,
        image_features: Optional[mx.array],
        video_features: Optional[mx.array],
    ) -> Optional[mx.array]:
        if image_features is None:
            return video_features
        if video_features is None:
            return image_features
        if len(image_features) != len(video_features):
            raise ValueError("image and video deepstack feature counts must match")

        token_rows = input_ids.tolist()
        merged_layers = []
        for image_layer, video_layer in zip(image_features, video_features):
            rows = []
            image_index = video_index = 0
            for token_row in token_rows:
                for token in token_row:
                    if token == self.config.image_token_index:
                        rows.append(image_layer[image_index])
                        image_index += 1
                    elif token == self.config.video_token_index:
                        rows.append(video_layer[video_index])
                        video_index += 1
            merged_layers.append(mx.stack(rows))
        return mx.stack(merged_layers)

    def hidden_state_at_layer(
        self,
        input_ids: mx.array,
        layer: int,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """Return the pre-final-norm hidden state after ``layer`` decoder layers."""
        features = self.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        return self.language_model.model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            mask=mask,
            position_ids=features.position_ids,
            visual_pos_masks=features.visual_pos_masks,
            deepstack_visual_embeds=features.deepstack_visual_embeds,
            stop_after_layer=layer,
            apply_final_norm=False,
        )

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, image_token_index, video_token_index
    ):
        special_image_mask = input_ids == image_token_index
        special_video_mask = input_ids == video_token_index
        special_image_mask = special_image_mask | special_video_mask
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask[..., None]
        special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)

        n_image_features = image_features.shape[0]
        n_image_mask_elements = special_image_mask.sum()
        if n_image_mask_elements != image_features.size:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        inputs_embeds = masked_scatter(
            inputs_embeds, special_image_mask, image_features
        )

        return inputs_embeds, special_image_mask

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):

        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )

        kwargs.update(
            {
                "pixel_values": pixel_values,
                **input_embeddings_features.to_dict(),
            }
        )

        logits = self.language_model(input_ids, mask=mask, cache=cache, **kwargs)
        return logits

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if "model" in key:
                if "model.language_model" in key:
                    key = key.replace("model.language_model", "language_model.model")

                elif "model.visual" in key:
                    key = key.replace("model.visual", "vision_tower")
            elif "lm_head" in key:
                key = key.replace("lm_head", "language_model.lm_head")

            sanitized_weights[key] = value

        return sanitized_weights

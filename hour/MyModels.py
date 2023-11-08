"""
### 참고
- https://keras.io/keras_core/api/layers/attention_layers/multi_head_attention/
- This is an implementation of multi-headed attention as described in the paper "Attention is all you Need" Vaswani et al., 2017.
"""

"""
Our model processes a tensor of shape (batch size, sequence length, features), 
where sequence length is the number of time steps and features is each input timeseries.
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K


"""
We include residual connections, layer normalization, and dropout. 
The resulting layer can be stacked multiple times.
The projection layers are implemented through keras.layers.Conv1D.
"""
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    LayerNormEps = 1e-6
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=LayerNormEps)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=LayerNormEps)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


"""
The main part of our model is now complete. We can stack multiple of those transformer_encoder blocks 
and we can also proceed to add the final Multi-Layer Perceptron classification head.
Apart from a stack of Dense layers, we need to reduce the output tensor of the TransformerEncoder part 
of our model down to a vector of features for each data point in the current batch. 
A common way to achieve this is to use a pooling layer. For this example,
a GlobalAveragePooling1D layer is sufficient.
"""
def build_transformer_model( # oritinal model
    input_shape,
    num_outputs,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units, 
    dropout=0,
    mlp_dropout=0,
    ):
    
    # 인코더
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    # 디코더
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu",
                         kernel_initializer='random_normal',
                         bias_initializer='zeros')(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    timeseries_outputs = layers.Dense(num_outputs, activation="relu",
                                      kernel_initializer='random_normal',
                                      bias_initializer='zeros')(x) # 양수만 나와야 하니까, linear 대신 relu
    return keras.Model(inputs, timeseries_outputs)


def build_transformer_conditionalInput_maxOutput_model_gAvgPooling(
    # 전력 PEAK를 조건부 입력으로 제공 + 출력층에서 시계열 데이터의 PEAK를 출력
    input_shape, conditional_input_shape, num_outputs,
    head_size, num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0, mlp_dropout=0,
    ):
    
    LayerNormEps = 1e-6
    """
    인코더 
    """
    inputs = keras.Input(shape=input_shape, name="ts_input") # 시계열 학습 데이터
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    # PEAK는 최대값을 예측하는 문제니까, max pooling이 더 좋으려나? 그렇지 않음.
    #x = layers.GlobalMaxPooling1D(data_format="channels_first")(x)
    #x = layers.MaxPooling1D(pool_size=4, strides=1, padding='same', data_format="channels_first")(x)
    
    # 조건부 입력(일간 총 사용량 예측치) 데이터를 추가로 입력 받음
    conditional_inputs = keras.Input(shape=conditional_input_shape, name="ts_max_input")
    # 조건부 입력(일간 총 사용량)의 단위가 크고, 다른 특징들은 이미 정규화가 되어있어서
    # 조건부 입력을 그대로 사용하면 이로 인한 업데이트가 너무나 과도해 질 수 있음
    # 따라서, 조건부 입력에 대해서도 정규화를 실시함
    normalized_conditional_inputs \
    = layers.LayerNormalization(epsilon=LayerNormEps)(conditional_inputs)
    # 인코더의 최종 출력 + 조건부 입력을 인코더 계층의 최종 출력으로 하고, 디코더로 전달
    x = x + normalized_conditional_inputs
    
    """
    디코더
    """
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu",
                         kernel_initializer='random_normal',
                         bias_initializer='zeros')(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    # 양수만 나와야 하니까, linear 대신 relu
    timeseries_outputs = layers.Dense(num_outputs, activation="relu",
                                      kernel_initializer='random_normal',
                                      bias_initializer='zeros',
                                      name='ts_output')(x) 
    
    #시계열 출력을 다시 입력으로 받아서, 출력의 element-wise max을 계산
    max_outputs = layers.Lambda(lambda v: K.max(v), 
                                output_shape=(1,1),
                                name="ts_max")(timeseries_outputs)
    # 최종 모델을 리턴
    return keras.Model(inputs = [inputs, conditional_inputs], 
                       outputs = [timeseries_outputs, max_outputs])
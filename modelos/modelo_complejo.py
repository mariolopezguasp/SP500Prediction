import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape, Multiply, Concatenate
from tensorflow.keras.regularizers import l2

def attention_block(inputs):
    """
    Bloque de atención simple.
    Calcula pesos sobre la dimensión temporal y los multiplica.
    """
    attention_probs = Dense(inputs.shape[-1], activation='softmax')(inputs)
    return Multiply()([inputs, attention_probs])

def build_complex_model(window_size, num_features, forecast_size=5, num_assets=12):
    """
    Construye y compila el modelo híbrido (Red Neuronal LSTM + Variable Macro).
    """
    # --- RAMA 1: Secuencial ---
    input_seq = Input(shape=(window_size, num_features), name="Input_Secuencial")
    lstm_1 = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(input_seq)
    dropout_1 = Dropout(0.2)(lstm_1)

    attention = attention_block(dropout_1)

    lstm_2 = LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))(attention)
    rama_secuencial = Dropout(0.2)(lstm_2)

    # --- RAMA 2: Macro (1 variable: FED) ---
    input_macro = Input(shape=(1,), name="Input_Macro_FED")
    # Le damos a la FED un peso en la red mediante una capa Dense
    rama_macro = Dense(8, activation='relu')(input_macro) 

    # --- FUSIÓN (Concatenación) ---
    fusion = Concatenate(name="Fusion_Capas")([rama_secuencial, rama_macro])

    # Capas finales de decisión
    densa_final = Dense(64, activation='relu')(fusion)
    salida_cruda = Dense(forecast_size * num_assets)(densa_final)
    outputs = Reshape((forecast_size, num_assets), name="Output_Prediccion")(salida_cruda)

    # Compilamos el modelo
    model_hibrido = Model(inputs=[input_seq, input_macro], outputs=outputs)
    model_hibrido.compile(optimizer='adam', loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
    
    return model_hibrido

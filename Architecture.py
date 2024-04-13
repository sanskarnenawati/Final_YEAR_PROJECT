import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('data.csv')
numeric = list(df.select_dtypes(include='float64').drop('sentiment', axis=1).columns)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['news'])
vocab_size = len(tokenizer.word_index) + 1

X_text = tokenizer.texts_to_sequences(df['news'])
X_text = tf.keras.preprocessing.sequence.pad_sequences(X_text)

def build_generator(latent_dim, vocab_size):
    model = Sequential([
        Embedding(vocab_size, latent_dim, input_length=X_text.shape[1]),
        LSTM(50),  # You can adjust the number of LSTM units as needed
        Dense(60, activation='tanh')  # Output size matches the OHLCV dimensions
    ])
    return model

# Define the Discriminator model
def build_discriminator():
    model = Sequential([
        Dense(50, input_shape=(60,)),  # 7 represents the OHLCV dimensions
        Dense(1, activation='sigmoid')
    ])
    return model

# Combine Generator and Discriminator into a GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = generator.input
    gan_output = discriminator(generator.output)
    model = Model(gan_input, gan_output)
    return model

# Define loss function
def gan_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

# Compile the models
latent_dim = 50  # Adjust as needed
generator = build_generator(latent_dim, vocab_size)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer=Adam(), loss='binary_crossentropy')

# Define training parameters
epochs = 100  # Number of epochs
batch_size = 32  # Batch size
half_batch = batch_size // 2  # Half batch size

# Training loop
for epoch in range(epochs):
    for batch in range(len(X_text) // batch_size):
        # Train discriminator
        # Select a random half-batch of OHLCV data
        idx = np.random.randint(0, len(X_text), half_batch)
        real_ohlcv = df[numeric].iloc[idx].values

        # Generate half-batch of synthetic OHLCV data
        random_index = np.random.randint(0, len(X_text), half_batch)
        random_text = X_text[random_index]
        generated_ohlcv = generator.predict(random_text)

        # Concatenate real and generated OHLCV data
        X_combined = np.concatenate([real_ohlcv, generated_ohlcv])

        # Labels for real and generated data
        y_combined = np.concatenate([np.ones((half_batch, 1)), np.zeros((half_batch, 1))])

        # Train discriminator
        d_loss = discriminator.train_on_batch(X_combined, y_combined)

        # Train generator
        # Generate noise
        noise = np.random.rand(batch_size, latent_dim)
        # Generate fake OHLCV data
        fake_labels = np.ones((len(random_text), 1))
        g_loss = gan.train_on_batch(random_text, fake_labels)

    # Print progress
    print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")


def generate_ohlcv(generator, text_input):
    text_input = tokenizer.texts_to_sequences([text_input])
    text_input = tf.keras.preprocessing.sequence.pad_sequences(text_input, maxlen=X_text.shape[1])
    generated_ohlcv = generator.predict(text_input)
    return generated_ohlcv


synthetic_ohlcv = generate_ohlcv(generator, "Your news headline goes here.")
print("Synthetic OHLCV:", synthetic_ohlcv)


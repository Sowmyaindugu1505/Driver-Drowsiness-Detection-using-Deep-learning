from tensorflow.keras.models import load_model

model = load_model("models/cnnCat2.h5")

# Print model structure
model.summary()

# Print weights of first layer
weights = model.layers[0].get_weights()
print(weights)

import numpy as np
import tf2onnx
import tensorflow as tf
from keras.src.layers import BatchNormalization
from onnx2pytorch import ConvertModel
import torch
import scipy.io
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.models import model_from_json

# Carica l'architettura del modello da un file JSON
with open('models/model.json', 'r') as json_file:
    model_json = json_file.read()

# Ricostruisci il modello
loaded_model = model_from_json(model_json,custom_objects={'BatchNormalization': BatchNormalization})

# Carica i pesi (se hai un file separato per i pesi, ad esempio 'model_weights.h5')
loaded_model.load_weights('models/weights.h5')
print('loaded model ok')

# Convert the model to ONNX format
spec = (tf.TensorSpec((None, *loaded_model.input_shape[1:]), tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(loaded_model, input_signature=spec)
model = onnx_model
for node in model.graph.node:
    if node.op_type == "Conv":
        for attr in node.attribute:
            if attr.name == "auto_pad" and attr.s == b'SAME_UPPER':
                attr.s = b'NOTSET'  # Disattiva auto_pad

# Convert ONNX model to PyTorch
pytorch_model = ConvertModel(onnx_model)
print(pytorch_model)
print('conversion ok')
# save
torch.save(pytorch_model.state_dict(), "models/pytorch_model.pth")

# Carica il file .mat
data = scipy.io.loadmat("test/data_test.mat")

xtest = np.zeros((data['CWT_ppg_test'].shape[1], data['CWT_ppg_test'][0,0]['cfs'][0,0].shape[0], data['CWT_ppg_test'][0,0]['cfs'][0,0].shape[1],2))
ytest = np.zeros((data['CWT_bp_test'].shape[1], data['CWT_bp_test'][0,0]['cfs'][0,0].shape[0], data['CWT_bp_test'][0,0]['cfs'][0,0].shape[1],2))

for i in range(data['CWT_ppg_test'].shape[1]):
    xtest[i,:,:,0] = np.real(data['CWT_ppg_test'][0,i]['cfs'][0,0])
    xtest[i,:,:,1] = np.imag(data['CWT_ppg_test'][0,i]['cfs'][0,0])
    ytest[i,:,:,0] = np.real(data['CWT_bp_test'][0,i]['cfs'][0,0])
    ytest[i,:,:,1] = np.imag(data['CWT_bp_test'][0,i]['cfs'][0,0])

# Converti i dati numpy in tensori PyTorch
xtest_tensor = torch.tensor(xtest, dtype=torch.float32)
ytest_tensor = torch.tensor(ytest, dtype=torch.float32)
print('data loaded ok')
# Verifica le dimensioni dei tensori
print("Dimensione xtest_tensor:", xtest_tensor.shape)
print("Dimensione ytest_tensor:", ytest_tensor.shape)

# Metti il modello in modalità di inferenza
pytorch_model.eval()

# Riduci il batch size a 1
xtest_tensor_single = xtest_tensor[0:1, :, :, :]
ytest_tensor_single = ytest_tensor[0:1, :, :, :]

# Se ci sono troppi canali, prendine solo 2
xtest_tensor_single_padded = xtest_tensor_single[:, :, :, :2]

# Verifica la nuova forma del tensore
print("Forma finale dell'input:", xtest_tensor_single_padded.shape)

# Esegui il modello con batch size 1
with torch.no_grad():
    predictions = pytorch_model(xtest_tensor_single_padded)

print("Predizioni:", predictions)

# Calcola l'errore quadratico medio (MSE)
mse = mean_squared_error(ytest_tensor.numpy(), predictions.numpy())

print("Errore quadratico medio (MSE):", mse)
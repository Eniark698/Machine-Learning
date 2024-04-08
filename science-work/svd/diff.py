import pickle
import matplotlib.pyplot as plt



with open('./temp/model_history_svd.pkl', 'rb') as file_pi:
    history_loaded_svd = pickle.load(file_pi)


with open('./temp/model_history_flat.pkl', 'rb') as file_pi:
    history_loaded_flat = pickle.load(file_pi)





fig, axs = plt.subplots(1, 2, figsize=(10, 5))

train_color = '#00bfff'  # Deep sky blue
val_color = '#ff7f50'    # Coral
train_color1 = '#3cb371'  # Medium Sea Green
val_color1 = '#ba55d3'    # Medium Orchid


axs[0].plot(history_loaded_svd['accuracy'], color=train_color,label=f'With SVD(train)')
axs[0].plot(history_loaded_svd['val_accuracy'], color=val_color,label=f'With SVD(test)')
axs[0].plot(history_loaded_flat['accuracy'], color=train_color1,label=f'Without SVD(train)')
axs[0].plot(history_loaded_flat['val_accuracy'], color=val_color1,label=f'Without SVD(test)')
axs[0].set_title('Model accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(loc='lower right')


axs[1].plot(history_loaded_svd['loss'], color=train_color,label=f'With SVD(train)')
axs[1].plot(history_loaded_svd['val_loss'], color=val_color,label=f'With SVD(test)')
axs[1].plot(history_loaded_flat['loss'], color=train_color1,label=f'Without SVD(train)')
axs[1].plot(history_loaded_flat['val_loss'], color=val_color1,label=f'Without SVD(test)')
axs[1].set_title('Model loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(loc='upper right')

plt.show()

import matplotlib.pyplot as plt

def plot_model(model):
    fig, axs = plt.subplots(1,2, figsize=(15,5))
    axs[0].plot(range(1,len(model.history['acc'])+1), model.history['acc'])
    axs[0].plot(range(1,len(model.history['val_acc'])+1), model.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')
    
    axs[1].plot(range(1,len(model.history['loss'])+1), model.history['loss'])
    axs[1].plot(range(1,len(model.history['val_loss'])+1), model.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
def plot_error_model(model):
    fig, axs = plt.subplots(1,2, figsize=(15,5))
    axs[0].plot(range(1,len(model.history['mse'])+1), model.history['mse'])
    axs[0].plot(range(1,len(model.history['val_mse'])+1), model.history['val_mse'])
    axs[0].set_title('Model Mean Standard Error')
    axs[0].set_ylabel('Error')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')
    
    axs[1].plot(range(1,len(model.history['mae'])+1), model.history['mae'])
    axs[1].plot(range(1,len(model.history['val_mae'])+1), model.history['val_mae'])
    axs[1].set_title('Model Mean Average Error')
    axs[1].set_ylabel('Error')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
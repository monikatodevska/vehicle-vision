"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 09.03.2021

Description: function library
             training process monitoring and result statistics
Python version: 3.6
"""

# python imports
import matplotlib.pyplot as plt
import os
import numpy as np

from tensorflow.keras.callbacks import Callback

class ValidationLossCallback(Callback):
    def __init__(self, validation_generator):
        self.validation_generator = validation_generator

    def on_epoch_end(self, epoch, logs=None):
        val_loss = self.model.evaluate_generator(self.validation_generator, verbose=0)
        print(f'Validation loss: {val_loss}')



class TrainingPlot(Callback):
    """
    Keras callback to enable live plot of training and validation loss that updates after every epoch
    To select whether to plot loss or accuracy, uncomment the corresponding block below. Plotting both in not enabled.
    To use:
        1. plot_losses = helper_stats.TrainingPlot()
        2. add plot_losses to the list of callbacks in model.fit
    Takes no input parameters. (For figure saving option, uncomment __init__ method and supply destination path in constructor.)
    """

    def __init__(self, dst_path_class_train,dst_path_class_val,dst_path_reg_train,dst_path_reg_val):

        self.dst_path_class_train = dst_path_class_train    # instance variable, unique to each instance
        self.dst_path_class_val = dst_path_class_val    # instance variable, unique to each instance
        self.dst_path_reg_train=dst_path_reg_train
        self.dst_path_reg_val=dst_path_reg_val

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses_class_train = []
        self.losses_reg_train=[]
        self.losses_class_val=[]
        self.losses_reg_val=[]

        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses_class_train.append(logs.get('loss'))
        self.losses_reg_train.append((logs.get('out_reg_loss')))


        self.losses_class_val.append(logs.get('val_loss'))
        self.losses_reg_val.append((logs.get('val_out_reg_loss')))


        # self.losses.append(logs.get('loss'))
        # self.acc.append(logs.get('accuracy'))
        # self.val_losses.append(logs.get('val_loss'))
        # self.val_acc.append(logs.get('val_accuracy'))

        # Before plotting, ensure at least 2 epochs have passed
        if len(self.losses_class_train) >= 1:

            # N = np.arange(0, len(self.losses))

            # To chose the style of your preference
            # print(plt.style.available) to see the available options
            # plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed

            f=open(self.dst_path_class_train, 'a')
            f.write(str(self.losses_class_train[-1])+'\n')

            f.close()

            f=open(self.dst_path_class_val,'a')
            f.write(str(self.losses_class_val[-1])+'\n')
            f.close()

            f = open(self.dst_path_reg_train, 'a')
            f.write(str(self.losses_reg_train[-1])+'\n')
            f.close()

            f = open(self.dst_path_reg_val, 'a')
            f.write(str(self.losses_reg_val[-1])+'\n')
            f.close()



            '''
            # --- plot loss --- 
            plt.figure(num=1)   # add figure number to stop generating new figure after each epoch
            plt.plot(N, self.losses, 'g', label = "train_loss")     # green line
            plt.plot(N, self.val_losses, 'r', label = "val_loss")   # red line

            plt.title("Training and Validation Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            # plt.legend()      # legend is not used because it is expanded after each call

            plt.pause(0.05)     # to keep figure on screen, otherwise block=False removes the figure instantly
            plt.show(block=False)   # block=False stops the shown figure to block execution until closed

            # Make sure the output directory exists
            # plt.savefig(os.path.join(self.dst_path, 'train_epoch-{}.png'.format(epoch)))
            '''

            # --- plot accuracy ---
            # plt.figure(num=1)     # add figure number to stop generating new figure after each epoch
            # plt.plot(N, self.acc, 'g', label="train_acc")   # green line
            # plt.plot(N, self.val_acc, 'r', label="val_acc")     # red line
            #
            # plt.title("Training and Validation Accuracy [Epoch {}]".format(epoch))
            # plt.xlabel("Epoch #")
            # plt.ylabel("Accuracy")
            # plt.legend()      # legend is not used because it is expanded after each call
            #
            # plt.pause(0.05)     # to keep figure on screen, otherwise block=False removes the figure instantly
            # plt.show(block=False)   # block=False stops the shown figure to block execution until closed

            # Make sure the output directory exists
            # plt.savefig(os.path.join(self.dst_path, 'val_epoch-{}.png'.format(epoch)))




def save_training_logs(history, dst_path):
    """
    saves loss and accuracy training curves of training and validation dataset
    NOTE: names of items in history object may differ in different Keras versions
    :param history: Keras callback object which stores accuracy information in each epoch [Keras history object]
    :param dst_path: destination for the graph images
    :return: None
    """

    # --- save accuracy graphs of training and validation sets ---
    plt.plot(history.history['accuracy'], 'r')  # training accuracy
    plt.plot(history.history['val_accuracy'], 'g')  # validtaion accuracy
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid()
    # plt.show()    # blocks execution until figure is closed
    plt.savefig(os.path.join(dst_path, 'acc.png'))      # acc.png - name of accuracy graph
    plt.close()

    # --- save loss graphs of training and validation sets ---
    plt.plot(history.history['loss'], 'r')  # training loss
    plt.plot(history.history['val_loss'], 'g')  # validation loss
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    # plt.show()    # blocks execution until figure is closed
    plt.savefig(os.path.join(dst_path, 'loss.png'))     # loss.png - name of loss graph
    plt.close()

    # --- save loss and accuracy of training and validation sets as a txt file ---
    losses = np.column_stack((history.history['loss'], history.history['val_loss']))
    np.savetxt(os.path.join(dst_path, 'loss.txt'), losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")

    accuracies = np.column_stack((history.history['accuracy'], history.history['val_accuracy']))
    np.savetxt(os.path.join(dst_path, 'acc.txt'), accuracies, fmt='%.4f', delimiter='\t', header="TRAIN_ACC\tVAL_ACC")

def save_training_logs_ssd(history, dst_path,vehicles):
    """
    saves graphs for the loss and accuracy of both the training and validation dataset throughout the epochs for comparison
    :param history: Keras callback object which stores accuracy information in each epoch [Keras history object]
    :param dst_path: destination for the graph images
    :return: None
    """

    # --- save combined loss graph of training and validation sets ---
    plt.figure()
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'g')
    plt.title('Combined loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(dst_path, 'joint_loss.png'))
    plt.close()

    # --- save accuracy graphs of training and validation sets ---
    plt.plot(history.history['accuracy'], 'r')  # training accuracy
    plt.plot(history.history['val_accuracy'], 'g')  # validtaion accuracy
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid()
    # plt.show()    # blocks execution until figure is closed
    plt.savefig(os.path.join(dst_path, 'acc.png'))  # acc.png - name of accuracy graph
    plt.close()



    # --- save classification loss graph of training and validation sets ---
    if vehicles:
        plt.figure()
        plt.plot(history.history['out_class_loss'], 'r')
        plt.plot(history.history['val_out_class_loss'], 'g')
        plt.title('Classification loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_class', 'val_class'], loc='upper right')
        plt.grid()
        # plt.show()
        plt.savefig(os.path.join(dst_path, 'classification_loss.png'))
        plt.close()

        # --- save regression loss graph of training and validation sets ---
        plt.figure()
        plt.plot(history.history['out_reg_loss'], 'r')
        plt.plot(history.history['val_out_reg_loss'], 'g')
        plt.title('Regression loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_reg', 'val_reg'], loc='upper right')
        plt.grid()
        # plt.show()
        plt.savefig(os.path.join(dst_path, 'regression_loss.png'))
        plt.close()

    # --- save losses of training and validation sets as txt files ---
    joint_losses = np.column_stack((history.history['loss'], history.history['val_loss']))
    np.savetxt(os.path.join(dst_path, 'joint_loss.txt'), joint_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")
    if vehicles:
        class_losses = np.column_stack((history.history['out_class_loss'], history.history['val_out_class_loss']))
        np.savetxt(os.path.join(dst_path, 'classification_loss.txt'), class_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")

        reg_losses = np.column_stack((history.history['out_reg_loss'], history.history['val_out_reg_loss']))
        np.savetxt(os.path.join(dst_path, 'regression_loss.txt'), reg_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")

# def save_training_logs_ssd(history, dst_path):
#     """
#     saves loss training curve of training and validation dataset
#     saves separate graphs of classification, regression, and joint loss
#     NOTE: names of items in history object may differ in different Keras versions
#     :param history: Keras callback object which stores loss and accuracy information in each epoch [Keras history object]
#     :param dst_path: destination for the graph images
#     :return: None
#     """
#
#     # --- save joint loss graph of training and validation sets ---
#     plt.figure()
#     plt.plot(history.history['loss'], 'r')  # training loss
#     plt.plot(history.history['val_loss'], 'g')  # validation loss
#     plt.title('Combined loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train_joint', 'val_joint'], loc='upper right')
#     plt.grid()
#     # plt.show()    # blocks execution until figure is closed
#     plt.savefig(os.path.join(dst_path, 'joint_loss.png'))
#     plt.close()
#
#     # --- save accuracy ---
#     # --- save classification loss graph of training and validation sets ---
#     plt.figure()
#     plt.plot(history.history['acc'], 'r')    # training loss
#     plt.plot(history.history['val_acc'], 'g')    # validation loss
#     plt.title('Classification accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train_class', 'val_class'], loc='lower right')
#     plt.grid()
#     # plt.show()    # blocks execution until figure is closed
#     plt.savefig(os.path.join(dst_path, 'classification_loss.png'))
#     plt.close()
#
#     # --- save classification loss graph of training and validation sets ---
#     plt.figure()
#     plt.plot(history.history['out_class_loss'], 'r')    # training loss
#     plt.plot(history.history['val_out_class_loss'], 'g')    # validation loss
#     plt.title('Classification loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train_class', 'val_class'], loc='upper right')
#     plt.grid()
#     # plt.show()    # blocks execution until figure is closed
#     plt.savefig(os.path.join(dst_path, 'classification_loss.png'))
#     plt.close()
#
#     # --- save regression loss graph of training and validation sets ---
#     plt.figure()
#     plt.plot(history.history['out_reg_loss'], 'r')  # training loss
#     plt.plot(history.history['val_out_reg_loss'], 'g')  # validation loss
#     plt.title('Regression loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train_reg', 'val_reg'], loc='upper right')
#     plt.grid()
#     # plt.show()    # blocks execution until figure is closed
#     plt.savefig(os.path.join(dst_path, 'regression_loss.png'))
#     plt.close()
#
#     # --- save losses of training and validation sets as txt files ---
#     joint_losses = np.column_stack((history.history['loss'], history.history['val_loss']))
#     np.savetxt(os.path.join(dst_path, 'joint_loss.txt'), joint_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")
#
#     class_losses = np.column_stack((history.history['out_class'], history.history['val_out_class_loss']))
#     np.savetxt(os.path.join(dst_path, 'classification_loss.txt'), class_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")
#     #
#     # reg_losses = np.column_stack((history.history['out_reg_loss'], history.history['val_out_reg_loss']))
#     # np.savetxt(os.path.join(dst_path, 'regression_loss.txt'), reg_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")
#

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

    return fig
#
#
# class TrainingPlot(Callback):
#     """
#     Keras callback to enable live plot of training and validation loss that updates after every epoch
#     To select whether to plot loss or accuracy, uncomment the corresponding block below. Plotting both in not enabled.
#     To use:
#         1. plot_losses = helper_stats.TrainingPlot()
#         2. add plot_losses to the list of callbacks in model.fit
#     Takes no input parameters. (For figure saving option, uncomment __init__ method and supply destination path in constructor.)
#     """
#
#     '''
#     def __init__(self, dst_path):
#         self.dst_path = dst_path    # instance variable, unique to each instance
#     '''
#
#     # This function is called when the training begins
#     def on_train_begin(self, logs={}):
#         # Initialize the lists for holding the logs, losses and accuracies
#         self.losses = []
#         self.acc = []
#         self.val_losses = []
#         self.val_acc = []
#         self.logs = []
#
#     # This function is called at the end of each epoch
#     def on_epoch_end(self, epoch, logs={}):
#
#         # Append the logs, losses and accuracies to the lists
#         self.logs.append(logs)
#         self.losses.append(logs.get('loss'))
#         self.acc.append(logs.get('accuracy'))
#         self.val_losses.append(logs.get('val_loss'))
#         self.val_acc.append(logs.get('val_accuracy'))
#
#         # Before plotting, ensure at least 2 epochs have passed
#         if len(self.losses) > 1:
#
#             N = np.arange(0, len(self.losses))
#
#             # To chose the style of your preference
#             # print(plt.style.available) to see the available options
#             # plt.style.use("seaborn")
#
#             # Plot train loss, train acc, val loss and val acc against epochs passed
#
#             '''
#             # --- plot loss ---
#             plt.figure(num=1)   # add figure number to stop generating new figure after each epoch
#             plt.plot(N, self.losses, 'g', label = "train_loss")     # green line
#             plt.plot(N, self.val_losses, 'r', label = "val_loss")   # red line
#
#             plt.title("Training and Validation Loss [Epoch {}]".format(epoch))
#             plt.xlabel("Epoch #")
#             plt.ylabel("Loss")
#             # plt.legend()      # legend is not used because it is expanded after each call
#
#             plt.pause(0.05)     # to keep figure on screen, otherwise block=False removes the figure instantly
#             plt.show(block=False)   # block=False stops the shown figure to block execution until closed
#
#             # Make sure the output directory exists
#             # plt.savefig(os.path.join(self.dst_path, 'train_epoch-{}.png'.format(epoch)))
#             '''
#
#             # --- plot accuracy ---
#             plt.figure(num=1)     # add figure number to stop generating new figure after each epoch
#             plt.plot(N, self.acc, 'g', label="train_acc")   # green line
#             plt.plot(N, self.val_acc, 'r', label="val_acc")     # red line
#
#             plt.title("Training and Validation Accuracy [Epoch {}]".format(epoch))
#             plt.xlabel("Epoch #")
#             plt.ylabel("Accuracy")
#             # plt.legend()      # legend is not used because it is expanded after each call
#
#             plt.pause(0.05)     # to keep figure on screen, otherwise block=False removes the figure instantly
#             plt.show(block=False)   # block=False stops the shown figure to block execution until closed
#
#             # Make sure the output directory exists
#             # plt.savefig(os.path.join(self.dst_path, 'val_epoch-{}.png'.format(epoch)))

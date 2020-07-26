
from mnist import *
from viz import *
import matplotlib as mpl
import matplotlib.pyplot as plt

# Uncomment if tex not installed

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{siunitx}',
#                                        r'\usepackage{amsmath}',
#                                        r'\usepackage{sansmathfonts}',
#                                        r'\usepackage[T1]{fontenc}',
#                                        r'\renewcommand*\familydefault{\sfdefault}']


epochs = 200 #200
batch_size = 512 #512
N_classes = 10
N = 10 ## Input size corresponding to the transformation

mnist_dp = MNISTDataProcessor()


# Initialize the data

data_DCT_N36 = mnist_dp.DCT(4)
#data_DCT_N36 = mnist_dp.fourier(3)

model_eo_DCT_N36 = construct_onn_EO_tf_ortho(N)
model_eo_DCT_N36.compile(optimizer='adam',
                 loss='mse',
                 metrics=['accuracy'])

#model_eo_DCT_N36.save('structure.h5')

history_eo_DCT_N36 = model_eo_DCT_N36.fit(data_DCT_N36.x_train,
                          data_DCT_N36.y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(data_DCT_N36.x_test, data_DCT_N36.y_test),
                          verbose=2)
model_eo_vars_DCT_N36 = [var.numpy() for var in model_eo_DCT_N36.variables]
model_eo_DCT_N36.save('./result10.h5', overwrite=True)
#model_eo_DCT_N36.save_weights('.\weight_36_fft.tf', overwrite=True)
# Initialize the models
# data_N36 = mnist_dp.fourier(3)
# model_eo_N36 = construct_onn_EO_tf_ortho(36)
# model_eo_N36.compile(optimizer='adam',
#                  loss='mse',
#                  metrics=['accuracy'])
#
#
# history_eo = model_eo_N36.fit(data_N36.x_train,
#                           data_N36.y_train,
#                           epochs=epochs,
#                           batch_size=batch_size,
#                           validation_data=(data_N36.x_test, data_N36.y_test),
#                           verbose=2)

#model_eo_vars = [var.numpy() for var in model_eo_N36.variables]



from tqdm import tqdm as pbar

possible_errors = [0.005 * i for i in range(20)]


def get_accuracies(model, model_vars, data):
    accuracies_for_errors = []
    accuracies_for_errors_t = []
    for e in pbar(possible_errors):
        for i in (0, 1, 2, 6, 7, 8):
            current_var = model_vars[i]
            model.variables[i].assign(current_var + e * np.random.randn(*current_var.shape))

        accuracies_for_errors.append(
            np.mean(tf.keras.metrics.categorical_accuracy(
                data.y_train, model.predict(data.x_train))))

        accuracies_for_errors_t.append(
            np.mean(tf.keras.metrics.categorical_accuracy(
                data.y_test, model.predict(data.x_test))))
    return accuracies_for_errors, accuracies_for_errors_t



accuracies_for_errors_DCT_N36, accuracies_for_errors_DCT_N36_t = get_accuracies(model_eo_DCT_N36, model_eo_vars_DCT_N36, data_DCT_N36)

iteration = epochs/5 # 40

## plot1
_, axes = plt.subplots(1, 2, figsize=(7, 3.5), dpi=300)

axes[0].plot(np.arange(iteration) * 5, history_eo_DCT_N36.history['accuracy'][::5], color=DARK_BLUE, linestyle=':')
axes[0].plot(np.arange(iteration) * 5, history_eo_DCT_N36.history['val_accuracy'][::5], color=DARK_GREEN, linestyle=':')
axes[0].set_xlabel('Epoch', fontsize=16)
axes[0].set_ylabel('Accuracy', fontsize=16)
axes[0].set_ylim((0.85, 1.01))
#axes[0].set_ylim((0.5, 1.01))
axes[0].set_title(r'(a)', loc='left', x=-.3, fontsize=18)
axes[0].set_title('MNIST training', fontsize=18)
axes[0].legend((r'Train', r'Test'), fontsize=11)
axes[0].tick_params(labelsize=14)

axes[1].plot(possible_errors, accuracies_for_errors_DCT_N36, linestyle=':', color=DARK_BLUE)
axes[1].plot(possible_errors, accuracies_for_errors_DCT_N36_t, linestyle=':', color=DARK_GREEN)

axes[1].set_xlabel(r'Phase Error ($\sigma_\theta, \sigma_\phi$)', fontsize=16)
axes[1].set_ylabel(r'Accuracy', fontsize=16)
axes[1].set_title(r'(b)', loc='left', x=-.3, fontsize=18)
axes[1].set_title('Robustness analysis', fontsize=18)
axes[1].set_ylim((0.64, 1.1))
#axes[1].set_ylim((0.5, 1))
axes[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[1].set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])

axes[1].tick_params(labelsize=14)

plt.tight_layout()
plt.show()
plt.savefig('nnanalysis_10.png', bbox_inches='tight')


##Plot2
std = np.sqrt(0.0025)
errors = []

inds_1 = [0, 1]
inds_2 = [6, 7]

for i in (0, 1, 2, 6, 7, 8):
    current_var = model_eo_vars_DCT_N36[i]
    errors.append(std * np.random.randn(*current_var.shape))

accuracies = []



for n in pbar(range(N * 2 + 1)):
    if n <= 64:
        for i in (0, 1):
            error_mask = np.zeros_like(model_eo_vars_DCT_N36[i])
            error_mask[n:] = 1
            model_eo_DCT_N36.variables[i].assign(model_eo_vars_DCT_N36[i] + errors[i] * error_mask)
    for i in (6, 7):
        if n >= 64:
            error_mask = np.zeros_like(model_eo_vars_DCT_N36[i])
            error_mask[n - N:] = 1
        else:
            error_mask = np.ones_like(model_eo_vars_DCT_N36[i])
        model_eo_DCT_N36.variables[i].assign(model_eo_vars_DCT_N36[i] + errors[i - 3] * error_mask)
    accuracies.append(
        np.mean(
            tf.keras.metrics.categorical_accuracy(data_DCT_N36.y_test, model_eo_DCT_N36.predict(data_DCT_N36.x_test))
        )
    )

_, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=300)

title_fontsize = 18

axes[0].plot(history_eo_DCT_N36.history['accuracy'], color=DARK_BLUE, linestyle='--')
axes[0].plot(history_eo_DCT_N36.history['val_accuracy'], color=DARK_GREEN, linestyle='--')
axes[0].set_xlabel('Epoch', fontsize=title_fontsize-2)
axes[0].set_ylabel('Accuracy', fontsize=title_fontsize-2)
axes[0].legend(['Train', 'Test'], fontsize=title_fontsize-2)
axes[0].set_ylim((0.8, 1))
#axes[0].set_ylim((0.5, 1))
axes[0].set_title('MNIST training',y=1.1, fontsize=title_fontsize)
axes[0].set_yticks([0.8, 0.84, 0.88, 0.92, 0.96, 1.0])
axes[0].tick_params(labelsize=14)

axes[1].plot(accuracies, color=DARK_GREEN)
axes[1].plot(np.ones_like(accuracies) * accuracies[-1], color=DARK_RED, linestyle='--')
axes[1].arrow(0, accuracies[-1], 0, accuracies[0] - accuracies[-1] +0.02, color=DARK_RED,
              head_width=5, head_length=0.02, length_includes_head=True)
axes[1].text(x=16, y=0.9, s='Drift', color=DARK_RED,
             horizontalalignment='center', verticalalignment='center', fontsize=16)
axes[1].text(x=64, y=0.7, s='Programming', color=DARK_ORANGE,
             horizontalalignment='center', verticalalignment='center', fontsize=16)
axes[1].arrow(0, 0.675, 128, 0, color=DARK_ORANGE,
              head_width=0.02, head_length=5, length_includes_head=True)
axes[1].set_xlabel('Number of programmed layers', fontsize=title_fontsize-2)
#axes[1].set_xticks([l * 32 for l in range(5)])
axes[1].set_yticks([0.7, 0.8, 0.9, 1.0])
axes[1].set_ylim((0.5, 1))
axes[1].set_ylabel('Accuracy', fontsize=title_fontsize-2)
axes[1].set_title('Parallel nullification',y=1.1, fontsize=title_fontsize)
axes[1].set_ylim((0.65, 1))
axes[1].tick_params(labelsize=14)
plt.tight_layout()
plt.show()
plt.savefig('nnanalysis2_10.png', bbox_inches='tight')
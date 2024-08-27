# Prosody_SincNet_CNN

This project will focus on the use of a convolutional neural network that has been specially trained for this case. After a thorough analysis of the language, the approach will be to provide an evaluation of nine common types of errors in English that have been carefully selected to assess each phrase the user pronounces. The goal is to develop an interface where users must read phrases in English and, by passing these phrases in real-time through our neural network, receive an automated and detailed evaluation of their performance. This approach aims to provide an effective tool to explain in a very personalized way what errors each person might have when speaking English, which can be crucial for their language learning.

To download the audios, the Jotform web gives all the completed forms in a single csv, containing all the links to download each individual audio. Descargar_audios.ipynb was made to automatically download all sentences into folders. Creating a folder for each sentence, where each folder has all the inputs of a specific sentence. Later, with
nombres_unicos.ipynb all audios were put into a single folder with unique names, for the commodity of the neuronal net. (As the audios weight 3Gb, if interested ask for it at matias.mussini@alumnos.upm.es)

The labeling of each one of the 1900 audios was handmade, the code clasificar audios.ipynb was made to agile the process of labeling. With this code, visual representation of the audio was shown among with sound, in order to define the presence or absence of each one of the nine defined errors.

When labeling was completed, one csv for each different sentence was created, resulting in 25 csvs. Inside each csv, nine labels are made for each one of the 76 participant’s input. Rellenar csv.ipynb was made to add the labels of the artificial intelligence generated answers. With vector_correcto.ipynb, a problem of data la-
beling was solved, dividing a column that was supposed to be two. Lastly, with nombres_unicos.ipynb, a column with the name of the file was added, in order to be able to recognise which label corresponds to which audio, this enabled the run of CrearUnSoloCsv.ipynb, which joined all csvs into a single one. 30 corrupt audios
were discarded as they could not be processed. Some audios had background noise but after an analysis it was decided to leave this audios as valid inputs, as the noise was not loud enough and the network will not be affected by it. This left the network with 1,870 audios and 16,830 labels.

All this process was made to facilitate the generation of X (audios) and y (labels) variables when creating the network.

With the great tool of SincNet, a convolutional neuronal network, made to analyze characteristics in raw data of one dimension, this tool and a 1D CNN was the chosen choice.

The audio files received from the forms required a final processing step to be ready for input into our model. These audios, obtained from Jotform, were in two channels. To represent the data in one dimension, a transformation was applied.

The audio files were received in three different sample rates: 48,000 Hz (1,844 files), 44,000 Hz (25 files), and 26,000 Hz (1 file). The network needs consistency in this area, so all files were standardized to 48,000 Hz, as it was the most common sample rate in the data. Although audio analysis can be effectively performed at 8,000 Hz or 16,000 Hz, the higher sample rate was maintained to take advantage of the high definition.

The network requires all data to be of the same length because it applies convolution, pooling, and ReLU layers, which expect uniform input sizes. To achieve this, padding was used to extend all audio files to match the length of the longest one, ensuring no data was lost. This padding only added the necessary amount of silence at the end of each audio file to reach the longest duration, which does not affect the audio quality. Additionally, the SincNet application in the initial layers of the model is designed to ignore silences at the beginning or end of sentences, so the padding does not impact the model’s performance.

When analyzed, the data seemed to already be normalized, as amplitude had a range of values between -1 and 1. A normalization was made even so, but when testes, almost all values were 0, so the original amplitude  of the audio files was used.

When the data was loaded into the model, it occupied 12 gigabytes in float64 format and 6 gigabytes in float32 format. Both sizes were too large for direct processing, necessitating the use of HDF5.

HDF5 was useful for storing large datasets on a local disk, due to its efficient data compression capabilities for partial data loading. This allowed to store the dataset on the local disk, in float 32 format, enabling the ability to read it as needed, without loading the entire dataset into memory at once, which is particularly beneficial for the problem that occurred when handling the large dataset.

Once the data is ready, the model was defined with the use of SincNet in its tensorflow implementation in the first layer, along with LeakyReLU, Flatten, MaxPooling1D, Input, Dropout, Dense, Conv1D functions, for the creation of each layer of convolution. Code can be found at model.ipynb

Once the model was built, it was saved at Prosodynet.h5 format. This allows the model weights to be imported by anyone with no need to run the previous code, using the keras function load_model. (As the model weights 6Gb, if interested ask for it at matias.mussini@alumnos.upm.es)

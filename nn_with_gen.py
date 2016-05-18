
from keras.preprocessing import text as TE
from keras.preprocessing.text import base_filter
import numpy as np
from numpy import log as Log
import sys
from keras.models import Model
from keras.layers import Dense, Dropout, merge, Input
import os
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax
import datetime
from keras.utils.visualize_util import plot
from keras.regularizers import l1l2
import csv
from keras import backend as K

from fields_config import categorial_fields, embbeding_fields, value_fields, sample_len_data_dict, \
    value_field_voc, validation_of_samples_length, get_data_train_det_dict, cat_field_voc
from Generator import Generator as Gen
'''
    Models's hyper parameters
'''
train_test_fraction = 0.8
train_validation_fraction = 0.9
delete_data_after_split = True
do_shuffle_on_data_when_split_train_test = True
repeat_vec_dict_config = {
    "do_repeat_vec": True, # If to repeat vector
    "num_of_times_to_repeat": 8,
    "on_this_field": 'Greengeeks_clicks',
    "on_this_value": '1'
    }

number_of_epochs = 15
data_on_ram = 10000
last_activation_function = 'sigmoid' # activation for the last layer
loss_function = 'binary_crossentropy' # options: 'mse', 'binary_crossentropy' , 'msle', 'mean_absolute_percentage_error'

# need to chose only one of the following updating method, 5 of them should be in comment
#optimizer_method = ["sgd", 0.001, 0.9, 1e-06, True] # [name_of_update_alg, lr(recomended:0.001), momentum(recommended:0.9), decay(recommended:1e-06), nesteruv(recommended:True)]
#optimizer_method = ["rmsprop", 0.001, 0.9, 1e-06] # [name_of_update_alg, lr(recomended:0.001), rho(recommended:0.9), epsilon(recommended:1e-06)]
#optimizer_method = ["adagrad", 0.01, 1e-06] # [name_of_update_alg, lr(recomended:0.01), epsilon(recommended:1e-6)]
#optimizer_method = ["adadelta", 1.0, 0.95, 1e-06] # [name_of_update_alg, lr(recomended:1.0), rho(recommended:0.95), epsilon(recommended:1e-06)]
#optimizer_method = ["adam", 0.001, 0.9, 0.999, 1e-08] # [name_of_update_alg, lr(recomended:0.001), beta_1(recommended:0.9), beta_2(recomanded:0.999) epsilon(recommended:1e-08)]
optimizer_method = ["adamax", 0.001, 0.9, 0.999, 1e-08] # [name_of_update_alg, lr(recomended:0.002), beta_1(recommended:0.9), beta_2(recomanded:0.999) epsilon(recommended:1e-08)]

l1_reglazation = 0.0000001
l2_reglazation = 0.0
do_shuffle_per_epoch = True
batch_size = 8
batch_size_for_evaluate = 1

dir_data='data_{D}'.format(D=str(datetime.datetime.now())[:10])

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
create_dir = True
try_ = 0
while create_dir:
    try_ += 1
    temp_dir = dir_data +"_{}/".format(try_)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        create_dir = False
        dir_data = temp_dir

path = os.path.dirname(os.path.abspath(__file__)) +'/'
# Creating log file
sys.stdout = Logger(path + dir_data + "log_data.txt".format(d=datetime.datetime.now()))

'''
    Printing hyper parameters config
'''
print ""
print "Net config for this run..."
print "num of epochs:", number_of_epochs
print "train/test fraction:", train_test_fraction
print "do shuffle on data per epoch:", do_shuffle_per_epoch
print "optimize method config:", optimizer_method
print "l1 regularization:", l1_reglazation
print "l2 regularization:", l2_reglazation
print "batch size:", batch_size
print "repeat vector config:", repeat_vec_dict_config
print "loss function:", loss_function

'''
    Working on data
'''
data_dict, train_dict, test_dict = get_data_train_det_dict(train_test_fraction=train_test_fraction,
                                                           do_shuffle_on_data_when_split_train_test=do_shuffle_on_data_when_split_train_test,
                                                           repeat_vec_dict_config=repeat_vec_dict_config
                                                           )
print ""
print "Working on data..."

class token(TE.Tokenizer):
    def __init__(self, field_name, nb_words=None, filters=base_filter(),
                 lower=True, split=' ', char_level=False, arr_of_text = []):
        self.text = ""
        self.field_name = field_name
        self.arr_of_text = arr_of_text
        super(token,self).__init__(nb_words, filters, lower, split, char_level)
        self.create_text()
        self.fit_on_texts([self.text])

    def create_text(self):
        for t in self.arr_of_text:
            self.text += str(t) + " "

'''
    Creating the vaocabolary dicts for value fields and categorial fields.
    Creating object of emmbeding classes for the emmbeding fields.
'''
voc_dict = {}
for field in categorial_fields:
    globals()["voc_dict_{F}".format(F=field)] = cat_field_voc(field, data_dict)
    voc_dict[field] = globals()["voc_dict_{F}".format(F=field)]

for field in value_fields:
    globals()["voc_dict_{F}".format(F=field)] = value_field_voc(field, data_dict)
    voc_dict[field] = globals()["voc_dict_{F}".format(F=field)]

for field in embbeding_fields:
    globals()["C_{F}".format(F=field)] = token(str(field), arr_of_text=data_dict[field])
    voc_dict[field] = globals()["C_{F}".format(F=field)]
    sample_len_data_dict[field] = len(globals()["C_{F}".format(F=field)].word_index) + 1

if delete_data_after_split:
    del data_dict

'''
    Starting building the net
'''
print "Building model..."

def add_input(filed_name, input_len_field, layers_name, total_len = 0):
    temp_len = int(np.ceil(Log(input_len_field)) + 1)
    # Input layer
    globals()["input_{F}".format(F=filed_name)] = Input(shape=(input_len_field,),
                                                        name="input_{F}".format(F=filed_name)
                                                        )
    # "Emmbeding" layer
    globals()["input_{F}_D".format(F=filed_name)] = Dense(temp_len,
                                                          activation='sigmoid',
                                                          init='uniform',
                                                          name="input_{F}_D".format(F=filed_name),
                                                          W_regularizer=l1l2(l1=l1_reglazation, l2=l2_reglazation)
                                                          )(globals()["input_{F}".format(F=filed_name)])
    layers_name.append(globals()["input_{F}_D".format(F=filed_name)])
    total_len[0] += temp_len

total_len=[int(0)]
layers_name=[]

for field in sample_len_data_dict:
    add_input(str(field), sample_len_data_dict[field], layers_name, total_len=total_len)

merge_l = merge(layers_name,
                mode='concat')

# first layer
L_1 = Dense(total_len[0]+20,
                W_regularizer=l1l2(l1=l1_reglazation, l2=l2_reglazation),
                activation='sigmoid',
                name="L_1"
            )(merge_l)

# second layer
L_2 = Dense(total_len[0]+20,
                W_regularizer=l1l2(l1=l1_reglazation, l2=l2_reglazation),
                activation='sigmoid',
                name="L_2"
            )(L_1)

# third layer
L_3 = Dense(total_len[0]+10,
                W_regularizer=l1l2(l1=l1_reglazation, l2=l2_reglazation),
                activation='sigmoid',
                name="L_3"
            )(L_2)

# output layer
output = Dense(1,
                W_regularizer=l1l2(l1=l1_reglazation, l2=l2_reglazation),
                activation= last_activation_function,
                name="output"
            )(L_3)

if optimizer_method[0] == "sgd":
    optimizer = SGD(lr=optimizer_method[1],
              decay=optimizer_method[2],
              momentum=optimizer_method[3],
              nesterov=optimizer_method[4])

if optimizer_method[0] == "rmsprop":
    optimizer = RMSprop(lr=optimizer_method[1],
                        rho=optimizer_method[2],
                        epsilon=optimizer_method[3])

if optimizer_method[0] == "adagrad":
    optimizer = Adagrad(lr=optimizer_method[1],
                        epsilon=optimizer_method[2])

if optimizer_method[0] == "adadelta":
    optimizer = Adadelta(lr=optimizer_method[1],
                         rho=optimizer_method[2],
                         epsilon=optimizer_method[3])

if optimizer_method[0] == "adam":
    optimizer = Adam(lr=optimizer_method[1],
                     beta_1=optimizer_method[2],
                     beta_2=optimizer_method[3],
                     epsilon=optimizer_method[4])

if optimizer_method[0] == "adamax":
    optimizer = Adamax(lr=optimizer_method[1],
                     beta_1=optimizer_method[2],
                     beta_2=optimizer_method[3],
                     epsilon=optimizer_method[4])




model = Model(input = [globals()["input_{F}".format(F=field)] for field in sample_len_data_dict],
                output = output)

model.compile(loss=loss_function,
              metrics={'output': 'accuracy'},
              optimizer=optimizer)

plot(model, to_file=path + dir_data + 'model.png', show_shapes=True)

print  model.get_config()

'''
    Start training the model
'''
print "Training..."
length_of_train_data = validation_of_samples_length(train_dict)
length_of_test_data = validation_of_samples_length(test_dict)

gen_for_train = Gen(train_dict, length_of_train_data, batch_size=batch_size, shuffle_per_epoch=do_shuffle_per_epoch, voc_dict=voc_dict)
model.fit_generator(
          gen_for_train.generator,
          length_of_train_data,
          nb_epoch=number_of_epochs,
          verbose=1,
          max_q_size=data_on_ram
          )

gen_for_test = Gen(test_dict, length_of_test_data, batch_size=batch_size_for_evaluate, shuffle_per_epoch=False, voc_dict=voc_dict)
score = model.evaluate_generator(
                       gen_for_test.generator,
                       length_of_test_data,
                       max_q_size=data_on_ram
                       )

print "The score is:", score[0]
print "The accuracy is:", score[1]

print ""
print('Generating submission...')

def generate_sample(data_dict, index):
    temp_batch_sample = {}
    for field in data_dict:
        if (field in categorial_fields) or (field in value_fields):
            temp_batch_sample["input_{F}".format(F=field)] = globals()["voc_dict_{F}".format(F=field)][data_dict[field][index]]
        elif field in embbeding_fields:
            temp_arr_sample = globals()["C_{F}".format(F=field)].texts_to_matrix([data_dict[field][index]])[0]
            temp_arr_sample = temp_arr_sample.reshape((1, sample_len_data_dict[field]))
            temp_batch_sample["input_{F}".format(F=field)] = temp_arr_sample
    return temp_batch_sample

#prob = K.function([model.input], [model.output])
def make_submission(test_dict, length_of_test_data, fname = "keras.csv"):
    with open(fname, 'wb') as f:
        a = csv.writer(f, delimiter=',')
        a.writerow(['id', 'predict_val', 'true_val'])
        for i in range(length_of_test_data):
            temp_id = test_dict['vt_id'][i]
            #temp_prob = np.float32(model.predict_on_batch(generate_sample(test_dict, i))[0][0])
            temp_prob = np.float64(model.predict(generate_sample(test_dict, i), batch_size=1, verbose=0)[0][0])
            #temp_prob = prob([generate_sample(test_dict, i)])
            temp_tr_val = test_dict['Greengeeks_clicks'][i]
            temp = [str(temp_id), temp_prob, str(temp_tr_val)]
            a.writerow(temp)
    print ('Wrote submission to file {}.'.format(fname))

make_submission(test_dict, length_of_test_data, fname= path + dir_data + 'submission_result.csv')







'''
    not relevant, little code to work on the first table.
    NO reason prosses this code along with the above code
'''



# other_fields = []
# other_fields_ind = []
# for field in all_fields:
#     if field not in fields:
#         other_fields.append(field)
#         other_fields_ind.append(all_fields[field])
#
# other_fields_ind.append(9)
# print other_fields_ind
# new_data = []
# #
# for line in data:
#     temp = [val for i, val in enumerate(line) if i not in other_fields_ind]
#     new_data.append(temp)
#
# with open('/home/shai/Desktop/keras_model_final_code_befofe_spliiting_data/good_data_final.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(new_data)





# good_data = []
# bad_data = []
#
# # good_data.append(all_fields)
# # bad_data.append(all_fields)
# first_line = True
# for l in data:
#     if first_line:
#         good_data.append(l)
#         bad_data.append(l)
#         first_line = False
#     else:
#         if (l[23] in ['None', 'none']) or (l[all_fields['keyword']] in ['#NAME?', 'None', 'none']):
#             bad_data.append(l)
#         else:
#             good_data.append(l)
#
# with open('/home/shai/Desktop/keras_model_final_code_befofe_spliiting_data/good_data.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(good_data)
#
# with open('/home/shai/Desktop/keras_model_final_code_befofe_spliiting_data/bad_data.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(bad_data)
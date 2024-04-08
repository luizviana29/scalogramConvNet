from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.data import AUTOTUNE
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Dense, AvgPool2D, Concatenate, Flatten
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras import Model

def MultiFilterModule(inputs, pool_size=(2,1),name_block='Block1'):
    BN=BatchNormalization(name=name_block+'_BN')(AvgPool2D(pool_size=pool_size,name=name_block+'_AvgPool')(inputs))
    conv_32 = Conv2D(24,(1,32),padding='same',strides=(1,1),activation='elu',kernel_initializer=VarianceScaling(mode='fan_avg', distribution='truncated_normal', seed=None))(BN)
    conv_64 = Conv2D(24,(1,64),padding='same',strides=(1,1),activation='elu',kernel_initializer=VarianceScaling(mode='fan_avg', distribution='truncated_normal', seed=None))(BN)
    conv_96 = Conv2D(24,(1,96),padding='same',strides=(1,1),activation='elu',kernel_initializer=VarianceScaling(mode='fan_avg', distribution='truncated_normal', seed=None))(BN)
    conv_128 = Conv2D(24,(1,128),padding='same',strides=(1,1),activation='elu',kernel_initializer=VarianceScaling(mode='fan_avg', distribution='truncated_normal', seed=None))(BN)
    conv_192 = Conv2D(24,(1,192),padding='same',strides=(1,1),activation='elu',kernel_initializer=VarianceScaling(mode='fan_avg', distribution='truncated_normal', seed=None))(BN)
    conv_256 = Conv2D(24,(1,256),padding='same',strides=(1,1),activation='elu',kernel_initializer=VarianceScaling(mode='fan_avg', distribution='truncated_normal', seed=None))(BN)
    mid_conc=Concatenate(name=name_block+'_Concat')([conv_32,conv_64,conv_96,conv_128,conv_192,conv_256])
    bottleneck=Conv2D(36,(1,1),padding='same',strides=(1,1),activation='elu',kernel_initializer=VarianceScaling(mode='fan_avg', distribution='truncated_normal', seed=None))(mid_conc)
    return bottleneck

Input_Scal = Input(shape=(40,256,1),name='Escalograma')
conv1 = Conv2D(16,(1,5),padding='same',strides=(1,1),activation='elu',kernel_initializer=VarianceScaling(mode='fan_avg', distribution='truncated_normal', seed=None))(BatchNormalization(name='BN_Input')(Input_Scal))
conv2 = Conv2D(16,(1,5),padding='same',strides=(1,1),activation='elu',kernel_initializer=VarianceScaling(mode='fan_avg', distribution='truncated_normal', seed=None))(BatchNormalization(name='BN_Conv1')(conv1))
conv3 = Conv2D(16,(1,5),padding='same',strides=(1,1),activation='elu',kernel_initializer=VarianceScaling(mode='fan_avg', distribution='truncated_normal', seed=None))(BatchNormalization(name='BN_Conv2')(conv2))

bottleneck_1=MultiFilterModule(conv3, pool_size=(5,1),name_block='Block1')
bottleneck_2=MultiFilterModule(bottleneck_1, pool_size=(2,1),name_block='Block2')
bottleneck_3=MultiFilterModule(bottleneck_2, pool_size=(2,1),name_block='Block3')
bottleneck_4=MultiFilterModule(bottleneck_3, pool_size=(2,1),name_block='Block4')


flat=Flatten(name='Flatten')(BatchNormalization(name='BN_flat')(bottleneck_4))
drop1=Dropout(0.5, name='Dropout')(flat)
dense_0=Dense(256,activation='elu', name='Dense0')(drop1)
dense_01=Dense(128,activation='elu', name='Dense01')(dense_0)
dense_1=Dense(64,activation='elu', name='Dense1')(dense_01)
dense_2=Dense(64,activation='elu', name='Dense2')(BatchNormalization(name='BN_dense1')(dense_1))

Output=Dense(140,activation='softmax',name='OutputBPM')(BatchNormalization(name='BN_Dense2')(dense_2))

ConvNet = Model(inputs=Input_Scal, outputs=Output)
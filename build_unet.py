from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam

def encoderBlock(input_layer,filters):
    c1 = Conv2D(filters,3,activation='relu',padding="same",input_shape=input_layer.shape)(input_layer)
    c2 = Conv2D(filters,3,activation='relu',padding="same",input_shape=input_layer.shape)(c1)
    m1 = MaxPooling2D(pool_size=2,padding='valid')(c2)
    return m1,c2

def decoderBlock(input_layer,filters,skip):
    t1 = Conv2DTranspose(filters,(2,2),strides=2, padding='valid')(input_layer)
    t1 = Concatenate()([t1,skip])
    c1 = Conv2D(filters,3,activation='relu',padding="same")(t1)
    c2 = Conv2D(filters,3,activation='relu',padding="same")(c1)
    return c2

def build_unet(input_shape,num_classes):

    Input_layer = Input(shape=input_shape)
    block1,skip1 = encoderBlock(Input_layer,64)
    block2,skip2 = encoderBlock(block1,128)
    block3,skip3 = encoderBlock(block2,256)
    block4,skip4 = encoderBlock(block3,512)

    final_encoder_block = Conv2D(1024,3,activation='relu',padding='same')(Conv2D(1024,3,activation='relu',padding='same')(block4))

    block5 = decoderBlock(final_encoder_block,512,skip4)
    block6 = decoderBlock(block5,256,skip3)
    block7 = decoderBlock(block6,128,skip2)
    block8 = decoderBlock(block7,64,skip1)

    if num_classes==1:
        activation="sigmoid"
    else:
        activation="softmax"

    output = Conv2D(filters=num_classes,kernel_size=1,activation=activation,padding='same')(block8)
    model = Model(Input_layer,output,name="U-Net")
    return model

# unet_model = build_unet((img_height,img_width,img_channels),1)
# unet_model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=1e-3),metrics=['accuracy'])
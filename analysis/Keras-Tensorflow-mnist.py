#!/usr/bin/env python
# coding: utf-8

#get_ipython().run_line_magic('matplotlib', 'inline')

import keras.datasets
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import keras.layers
import keras.layers.core
import keras.models

#plt.style.use('ggplot')


# 読み込みデータ件数
X_A = 60000
Y_A = 10000

# ピクセル（28×28）
NEW_PIX = 784

# y_trainが正解タグデータ
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# データの整形&正規化
X_train = X_train.reshape(X_A, NEW_PIX) / 255
X_test = X_test.reshape(Y_A, NEW_PIX) / 255


#idx = 0
#size = 28

#a, b = np.meshgrid(range(size), range(size))
#c = X_train[idx].reshape(size, size)
#c = c[::-1,:]

#print('描かれている数字: {}'.format(y_train[idx]))

#plt.figure(figsize=(2.5, 2.5))
#plt.xlim(0, 27)
#plt.ylim(0, 27)
#plt.tick_params(labelbottom="off")
#plt.tick_params(labelleft="off")
#plt.pcolor(a, b, c)
#plt.gray()


# ダミーコーディング
y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)


# DLのモデル作成
# Dense = 層 activation = 活性化関数
def modeling(hidd_batch_size,hidd_act,hidd_shape,drop,out_batch_size,out_act,com_loss,com_optimizer,com_metrics):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hidd_batch_size, activation = hidd_act, input_shape=(hidd_shape,)))
    model.add(keras.layers.core.Dropout(drop))
    model.add(keras.layers.Dense(out_batch_size, activation = out_act))
    model.compile(loss=com_loss, optimizer=com_optimizer, metrics=[com_metrics])
    return model

def score(model,fit_batch_size,fit_epochs):
    model.fit(X_train, y_train, batch_size=fit_batch_size, verbose=2, epochs=fit_epochs)
    score = model.evaluate(X_test, y_test, verbose=2)
    return score

# 隠れ層
hidd_batch_size = 512
hidd_act = 'sigmoid'
hidd_shape = 784
drop = 1

# 出力層(いくつかのカテゴライズを行う場合はsoftmaxと使う)
out_batch_size = 10
out_act = 'softmax'

# コンパイル
com_loss = 'categorical_crossentropy'
com_optimizer = 'sgd'
com_metrics = 'accuracy'

# モデルfit
fit_batch_size = 200
fit_epochs = 10

score1 = score(modeling(hidd_batch_size,hidd_act,hidd_shape,drop,
                        out_batch_size,out_act,
                        com_loss,com_optimizer,com_metrics),
               fit_batch_size,fit_epochs)[1]

# 活性化関数の変更 sigmoid → relu
hidd_act_new = 'relu'

score2 = score(modeling(hidd_batch_size,hidd_act_new,hidd_shape,drop,
                        out_batch_size,out_act,
                        com_loss,com_optimizer,com_metrics),
               fit_batch_size,fit_epochs)[1]

# 最適化関数の変更 sgd → adm
# https://keras.io/ja/optimizers/
com_optimizer_new = 'adam'

score3 = score(modeling(hidd_batch_size,hidd_act_new,hidd_shape,drop,
                        out_batch_size,out_act,
                        com_loss,com_optimizer_new,com_metrics),
               fit_batch_size,fit_epochs)[1]

# Dropout(汎化性能up/過学習防止)  1 → 0.2
drop_new = 0.2

score4 = score(modeling(hidd_batch_size,hidd_act_new,hidd_shape,drop_new,
                        out_batch_size,out_act,
                        com_loss,com_optimizer_new,com_metrics),
               fit_batch_size,fit_epochs)[1]


print("活性関数がシグモイド関数：" + str(score1)+
      "\n活性関数がLelu関数："+str(score2)+
      "\n最適化関数をadm関数：" + str(score3)+
      "\nドロップを設定：" +str(score4))

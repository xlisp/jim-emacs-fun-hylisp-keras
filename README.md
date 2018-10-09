# jim-emacs-fun-hylisp-keras 我的大脑从来不记忆公式,只是记忆书上不存在的Lisp,将哲学保存到每一个Lisp原子里面
* 从Hack计算代码到计算论文
* 用函数式LISP来表达问题,问题变得清晰很多
* 用李小龙和乔布斯的哲学推导吸引Hack计算代码: 首先你是个哲学家,然后才是一个Lisp程序员

##### hy2py3repl2
```bash
hy2py3repl2 () {
	rlwrap sh -c 'while read line; do pycode=`echo "$line" | hy2py3`; echo "翻译:"$pycode; echo "执行:"; python -c "print($pycode)"; echo "------------" ; done'
}
#  ------------
#  (-> 1 (+ 2) (- 1) (/ 4))
#  翻译:(1 + 2 - 1) / 4
#  执行:
#  0.5
#  ------------
```
##### import

```clojure
(import
 [tensorflow :as tf]
 keras
 [keras [backend :as K]]
 [keras.models [Model load_model]]
 [numpy :as np]
 [keras.layers [Input LSTM GRU Dense Embedding Bidirectional BatchNormalization Lambda]])
```

##### Input

```clojure
(Input :shape (, None) :name "Decoder-Input")
;;=> <tf.Tensor 'Decoder-Input_2:0' shape=(?, ?) dtype=float32>
(Input :shape (, 11) :name "Encoder-Input")
;;=> <tf.Tensor 'Encoder-Input_1:0' shape=(?, 11) dtype=float32>
```

##### summary

```clojure
((. decoder_model summary))
```

##### shape

```clojure
(import [keras.datasets [mnist]])
(setv (, (, train_images train_labels) (, test_images test_labels)) ((. mnist load_data)))

(. train_images shape) ;;=> (60000, 28, 28)
(len train_labels) ;;=> 60000
train_labels ;;=> array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)

(. test_images shape) ;;=> (10000, 28, 28)
(len test_labels) ;;=> 10000
test_labels ;;=> array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)

```

##### 神经网络的黑盒不黑get_layer & layers: 就像纯函数一样调用每个层或模型或映射或矩阵或函数
* get_layer
```clojure
;; 拆出来层当模型来用,黑盒映射的白盒化
(setv model_get_layer (fn [name] (-> model (.get_layer name))))

;;添加一个 x -> x^2 层
(model.add (Lambda (fn [x] (** x 2))))

(** (np.array [[[1 8] [3 5]] [[9 7] [6 4]]]) 2)
;; => array([[[ 1, 64],
;;         [ 9, 25]],
;;        [[81, 49],
;;         [36, 16]]])

(K.eval
 ((Lambda (fn [x] (** x 2)))
  (K.variable
   (np.array [[[1 8] [3 5]] [[9 7] [6 4]]]))))
;;=> <tf.Tensor 'lambda_8/pow:0' shape=(2, 2, 2) dtype=float32>
;;K.eval之后=>
;; array([[[  1.,  64.],
;;        [  9.,  25.]],
;;       [[ 81.,  49.],
;;        [ 36.,  16.]]], dtype=float32)
```
* layers
```clojure
(import [keras.applications.vgg16 [VGG16]]
        [keras.models [Model]]
        [keras.preprocessing [image]]
        [keras.applications.vgg16 [preprocess_input]]
        [numpy :as np])

;; github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
(setv base_model (VGG16 :weights "imagenet" :include_top True))

;; 先打印所有的layers出来看,以便get_layer单独取出层
(for [(, i layer) (enumerate base_model.layers)]
  (print i ": " layer.name ", " layer.input_shape ", " layer.output_shape))
;; 0 :  input_1 ,  (None, 224, 224, 3) ,  (None, 224, 224, 3)
;; 1 :  block1_conv1 ,  (None, 224, 224, 3) ,  (None, 224, 224, 64)
;; ...
;; 21 :  fc2 ,  (None, 4096) ,  (None, 4096)
;; 22 :  predictions ,  (None, 4096) ,  (None, 1000)

;;keras get weights of dense layer
(setv (, weights biases)
      (-> base_model (.get_layer "fc2") .get_weights))
;; weights=> weights.shape (4096, 4096)
;; biases=> biases.shape (4096,)
;; array([ 0.64710701,  0.48036072,  0.58551109, ...,  0.50245267,
;;         0.41782504,  0.66609925], dtype=float32)

;; 单个层的特征提取predict
(setv mmodel (Model :input base_model.input ;;<tf.Tensor 'input_2:0' shape=(?, 224, 224, 3) dtype=float32>
                    :output (-> base_model (.get_layer "block4_pool") (. output)) ;;<tf.Tensor 'block4_pool_1/MaxPool:0' shape=(?, 14, 14, 512) dtype=float32>
                    ))
(setv features
      (->
       "cat.jpg"
       (image.load_img :target_size (, 224 224))
       (image.img_to_array)
       (np.expand_dims :axis 0)
       (preprocess_input)
       (mmodel.predict)))

(-> features first len) ;;=> 14
(-> features (. shape)) ;; => (1, 14, 14, 512)
(-> features first first (. shape)) ;;=> (14, 512)

```
##### 步步为营保存层,层和模型嫁接迁移

```clojure
;; 保存层权重和numpy每一步结果,模型的每次结果

(np.save "max_emb_dim500_v2.npy" max_hs)

(seq2seq_Model.save "code_summary_seq2seq_model.h5")
(load_model "code_summary_seq2seq_model.h5")

(model.load_weights weights_path)
```

##### Dense softmax

```clojure
;; 密集连接(全连接):
;; 最后一层是一个14002路的softmax层, 返回一个由14002个概率值(总和为1)组成的数组,
;; 每个概率值表示 当前代码向量 属于14002个句向量类别中某一个的概率
;; keras.activations.softmax or K.softmax
(Dense 14002 :activation keras.activations.softmax :name "Final-Output-Dense")
(fn [data]
  (setv (, dec_bn2 _) data)
  ((model_get_layer "Final-Output-Dense") dec_bn2))
```

##### compile

```clojure
;; sparse_categorical_crossentropy 是整数(sparse)标签应该遵循分类编码
(seq2seq_Model.compile :optimizer (optimizers.Nadam :lr 0.00005)
                       :loss "sparse_categorical_crossentropy")
```
##### optimizer
```clojure
(model.compile :optimizer (SGD) :loss keras.losses.categorical_crossentropy)
```
##### predict黑盒映射

```clojure
(encoder_model.predict raw_tokenized)
(decoder_model.predict [state_value, encoding])
```
##### fit拟合数据

```clojure
(seq2seq_Model.fit [encoder_input_data decoder_input_data]
                   (np.expand_dims decoder_target_data, -1))
```

##### evaluate

```clojure
(setv (, test_loss test_acc) (network.evaluate test_images test_labels))
;;=> test_acc: 0.9785
```
##### get_weights & set_weights & load_weights & save_weights
```clojure
(->
 (tf.placeholder tf.float32 [None 10] :name "input_x")
 ;; OR: (K.placeholder [None 10] :name "input_x")
 ((fn [input_x]
    (setv dense1 (Dense 10 :activation K.relu))
    (dense1 input_x) ;;=> <tf.Tensor 'dense_2/Relu:0' shape=(?, 10) dtype=float32>
    dense1))
 (.get_weights)
 (first)
 len
 ) ;;=> 10
```
##### np.array张量0D~3D

```clojure
(import [numpy :as np])
(-> 12 np.array (. ndim)) ;; 0D, ()
(-> [1 3 5 8] np.array (. ndim)) ;;=> 1D, (4,)
(-> [[1 3] [5 8]] np.array (. ndim)) ;;=> 2D, (2, 2)
(-> [[[1 2] [3 5]] [[9 7] [6 4]]] np.array (. ndim)) ;;=> 3D , (2, 2, 2)
```

##### slice张量
```clojure
(-> train_images (get 4) (. shape)) ;;=> (28, 28)
(-> train_images (get (slice 10 100)) (. shape)) ;;=> (90, 28, 28)
;; 所有图像右下角选出14x14的像素区域
(-> train_images (get [(slice None) (slice 14 None) (slice 14 None)]) (. shape)) ;;=> (60000, 14, 14)
;; 所有图像中心裁剪出14x14的像素区域
(-> train_images (get [(slice None) (slice 7 -7) (slice 7 -7)]) (. shape)) ;;=> (60000, 14, 14)

```
##### 张量运算 AND OR (like 集合运算)
* 逐元素relu & add运算
```clojure
(K.eval
 (K.relu (np.array [[[1 8] [3 5]] [[9 7] [6 4]]])
         :alpha 0.0 :max_value 5))
;;=> <tf.Tensor 'Relu:0' shape=(2, 2, 2) dtype=int64>
;;K.eval => array([[[1, 5],
;;        [3, 5]],
;;       [[5, 5],
;;        [5, 4]]])

(np.add (np.array [[[1 8] [3 5]] [[9 7] [6 4]]])
        (np.array [[[2 8] [5 2]] [[9 8] [6 4]]]))
;; array([[[ 3, 16],
;;         [ 8,  7]],
;;        [[18, 15],
;;         [12,  8]]])
```
* maximum广播
```clojure
(np.maximum (np.array [[[1 8] [3 5]] [[9 7] [6 4]]]) 0.0)
;;array([[[ 1.,  8.],
;;        [ 3.,  5.]],
;;       [[ 9.,  7.],
;;        [ 6.,  4.]]])
```
* dot点积
```clojure
(np.dot (np.array [[[1 8] [3 5]] [[9 7] [6 4]]])
        (np.array [[[2 8] [5 2]] [[9 8] [6 4]]]))
;; array([[[[ 42,  24],
;;          [ 57,  40]],
;;         [[ 31,  34],
;;          [ 57,  44]]],
;;        [[[ 53,  86],
;;          [123, 100]],
;;         [[ 32,  56],
;;          [ 78,  64]]]])
```
* reshape张量编写
```clojure
(. (train_images.reshape (, 60000 (* 28 28))) shape) ;;=> (60000, 784)
(/ (train_images.astype "float32") 255)

(->
 [[0. 1.]
  [2. 3.]
  [4. 5.]]
 np.array
 (.reshape (, 2 3)))
;; array([[ 0.,  1.,  2.],
;;        [ 3.,  4.,  5.]])

(-> (np.zeros (, 300 200))
    np.transpose
    (. shape))
;; (200, 300)
```
* argmax
```clojure
(np.argmax (model.predict im))
```
* expand_dims
```clojure
(np.expand_dims im :axis 0)
```
* squeeze
```clojure
(np.squeeze out)
```
##### 分布式表示最大的问题在于: 连接connect的问题,连接多个层的问题,充分利用各种细胞的简单优势计算
* 就像你要获得pdf的C-g数据一样,不能像网页一样获得,但是pdf可以用苹果笔来写,网页却不能
```clojure

```
##### seq2seq model

```clojure

(defn build_seq2seq_model [word_emb_dim
                           hidden_state_dim
                           encoder_seq_len
                           num_encoder_tokens
                           num_decoder_tokens]
  (setv
   ;; Encoder Model
   encoder_inputs (Input :shape (, encoder_seq_len) :name "Encoder-Input")
   seq2seq_encoder_out
   (->> encoder_inputs
        ((Embedding num_encoder_tokens word_emb_dim :name "Body-Word-Embedding" :mask_zero False))
        ((BatchNormalization :name "Encoder-Batchnorm-1"))
        ((GRU hidden_state_dim :return_state True :name "Encoder-Last-GRU" :dropout 0.5))
        ((fn [gru_state]
           (setv (, _ state_h) gru_state)
           ((Model :inputs encoder_inputs :outputs state_h :name "Encoder-Model") encoder_inputs))))
   ;; Decoder Model
   decoder_inputs (Input :shape (, None) :name "Decoder-Input")

   seq2seq_Model
   (->> decoder_inputs
        ;; 生成Embedding高阶函数
        ((Embedding num_decoder_tokens word_emb_dim :name "Decoder-Word-Embedding" :mask_zero False))
        ((BatchNormalization :name "Decoder-Batchnorm-1"))
        ((fn [dec_bn]
           (setv (, decoder_gru_output _)
                 ((GRU hidden_state_dim :return_state True :return_sequences True :name "Decoder-GRU" :dropout 0.5)
                  dec_bn :initial_state seq2seq_encoder_out))
           decoder_gru_output))
        ((BatchNormalization :name "Decoder-Batchnorm-2"))
        ((Dense num_decoder_tokens :activation "softmax" :name "Final-Output-Dense"))
        ((fn [decoder_outputs]
           (Model [encoder_inputs decoder_inputs] decoder_outputs))))
   )
  seq2seq_Model
  )

```

##### extract model

```clojure
(defn extract_decoder_model [model]
  ;; Reconstruct the input into the decoder
  (setv model_get_layer (fn [name] (-> model (.get_layer name)))
        decoder_inputs (. (model_get_layer "Decoder-Input") input)
        gru_inference_state_input
        (Input :shape (, (get (. (model_get_layer "Encoder-Model") output_shape) -1))
               :name "hidden_state_input"))
  (setv decoder_model
        (-> decoder_inputs
            ((model_get_layer "Decoder-Word-Embedding"))
            ((model_get_layer "Decoder-Batchnorm-1"))
            ((fn [dec_bn]
               ((model_get_layer "Decoder-GRU") [dec_bn gru_inference_state_input])))
            ((fn [data]
               (setv (, gru_out gru_state_out) data)
               (, ((model_get_layer "Decoder-Batchnorm-2") gru_out) gru_state_out)))
            ((fn [data]
               (setv (, dec_bn2 gru_state_out) data)
               (, ((model_get_layer "Final-Output-Dense") dec_bn2) gru_state_out)))
            ((fn [data]
               (setv (, dense_out gru_state_out) data)
               (Model [decoder_inputs gru_inference_state_input]
                      [dense_out gru_state_out])))))
  decoder_model
  )

```

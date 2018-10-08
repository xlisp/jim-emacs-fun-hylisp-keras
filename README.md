# jim-emacs-fun-hylisp-keras

##### import

```clojure
(import
 [tensorflow :as tf]
 [keras [backend :as K]]
 [keras.models [Model load_model]]
 [keras.layers [Input LSTM GRU Dense Embedding Bidirectional BatchNormalization]])
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

##### get_layer

```clojure
;; 拆出来层当模型来用,黑盒映射的白盒化
(setv model_get_layer (fn [name] (-> model (.get_layer name))))
```

##### 步步为营保存层,层和模型嫁接迁移

```clojure
;; 保存层权重和numpy每一步结果,模型的每次结果

(np.save "max_emb_dim500_v2.npy" max_hs)

(seq2seq_Model.save "code_summary_seq2seq_model.h5")
(load_model "code_summary_seq2seq_model.h5")
```

##### Dense softmax

```clojure
;; 密集连接(全连接):
;; 最后一层是一个14002路的softmax层, 返回一个由14002个概率值(总和为1)组成的数组,
;; 每个概率值表示 当前代码向量 属于14002个句向量类别中某一个的概率
((Dense 14002 :activation "softmax" :name "Final-Output-Dense"))
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
        (Input :shape (, (get (. ((. model get_layer) "Encoder-Model") output_shape) -1))
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

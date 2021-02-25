from utils import *

get_session(gpu_fraction=0.3)
_flip_gradient = FlipGradientBuilder()

def create_placeholders(ht=28, wd=28, ch=3, classes=10):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    ht -- height of an input image
    wd -- width of an input image
    ch -- number of channels of the input
    classes -- number of classes, default = 10 (0, 1, ... 9)
        
    Returns:
    Xs, Xt -- data input for source/target tasks
    Ys, Yt -- input labels for source/target tasks
    D -- domain index (0 = source domain, 1 = target domain)
    """

    Xs = tf.placeholder(tf.float32, [None, ht, wd, ch], name='Xs')
    Xt = tf.placeholder(tf.float32, [None, ht, wd, ch], name='Xt')
    Ys = tf.placeholder(tf.float32, [None, classes], name='Ys')
    Yt = tf.placeholder(tf.float32, [None, classes], name='Yt')
    D = tf.placeholder(tf.float32, [None, 2], name='D')
    
    return Xs, Xt, Ys, Yt, D

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network

    Returns:
    parameters -- a dictionary of tensors containing weight parameters
    """
    
    Wf1 = tf.get_variable("Wf1", [3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Wf2 = tf.get_variable("Wf2", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    Wf3 = tf.get_variable("Wf3", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer(seed=2))
    
    bf1 = tf.get_variable("bf1", [32], initializer=tf.zeros_initializer())
    bf2 = tf.get_variable("bf2", [64], initializer=tf.zeros_initializer())
    bf3 = tf.get_variable("bf3", [64], initializer=tf.zeros_initializer())

    Wl1 = tf.get_variable("Wl1", [1024, 256], initializer=tf.contrib.layers.xavier_initializer(seed=4))
    Wl2 = tf.get_variable("Wl2", [256, 256], initializer=tf.contrib.layers.xavier_initializer(seed=5))
    Wl3 = tf.get_variable("Wl3", [256, 10], initializer=tf.contrib.layers.xavier_initializer(seed=6))

    bl1 = tf.get_variable("bl1", [256], initializer=tf.zeros_initializer())
    bl2 = tf.get_variable("bl2", [256], initializer=tf.zeros_initializer())
    bl3 = tf.get_variable("bl3", [10], initializer=tf.zeros_initializer())

    Wd1 = tf.get_variable("Wd1", [1024, 256], initializer=tf.contrib.layers.xavier_initializer(seed=7))
    Wd2 = tf.get_variable("Wd2", [256, 256], initializer=tf.contrib.layers.xavier_initializer(seed=8))
    Wd3 = tf.get_variable("Wd3", [256, 2], initializer=tf.contrib.layers.xavier_initializer(seed=9))

    bd1 = tf.get_variable("bd1", [256], initializer=tf.zeros_initializer())
    bd2 = tf.get_variable("bd2", [256], initializer=tf.zeros_initializer())
    bd3 = tf.get_variable("bd3", [2], initializer=tf.zeros_initializer())

    params = {"Wf1": Wf1, "Wf2": Wf2, "Wf3": Wf3,
                  "bf1": bf1, "bf2": bf2, "bf3": bf3,
                  "Wl1": Wl1, "Wl2": Wl2, "Wl3": Wl3,
                  "bl1": bl1, "bl2": bl2, "bl3": bl3,
                  "Wd1": Wd1, "Wd2": Wd2, "Wd3": Wd3,
                  "bd1": bd1, "bd2": bd2, "bd3": bd3
    }
    
    return params

def forward_propagation(X, params, modelType):
    """
    Forward propagation for the model
    
    Arguments:
    X -- input dataset placeholder
    params -- dictionary containing weight parameters defined in initialize_parameters
    modelType -- choose one among 'feature extractor', 'label predictor', 'domain classifier'

    Returns:
    Z -- the output of the last LINEAR unit
    """

    Z = X
    
    if modelType == 'feature extractor' :
        for i in range(3) :
            Z = tf.nn.conv2d(Z, params['Wf'+str(i+1)], strides=[1, 1, 1, 1], padding='SAME')
            Z = tf.nn.relu(Z + params['bf'+str(i+1)])
            Z = tf.nn.max_pool(Z, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
            Z = tf.nn.dropout(Z, rate=0.05)

        Z = tf.contrib.layers.flatten(Z)
    elif modelType == 'label predictor' : 
        for i in range(3) :
            Z = tf.matmul(Z, params['Wl'+str(i+1)]) + params['bl'+str(i+1)]

            if i == 2 :
                break

            Z = tf.nn.relu(Z)
            Z = tf.nn.dropout(Z, rate=0.05)
    elif modelType == 'domain classifier' : 
        for i in range(3) :
            Z = tf.matmul(Z, params['Wd'+str(i+1)]) + params['bd'+str(i+1)]

            if i == 2 :
                break

            Z = tf.nn.relu(Z)
            Z = tf.nn.dropout(Z, rate=0.05)
    else:
        print("You have passed a wrong argument\n",
              "Model type should be one among 'feature extractor', 'label predictor', 'domain classifier'")
        exit()

    return Z


class Model:
    """
    Creates a DANN model that will be trained on the MNIST (source) and MNIST-M (target) dataset
    """

    def __init__(self):
        self.build()

    def build(self):
        tf.reset_default_graph()
    
        Xs, Xt, Ys, Yt, D = create_placeholders(28, 28, 3, 10)
        l = tf.placeholder(tf.float32, [], name='l')        # Gradient reversal scaler
        params = initialize_parameters()
    
        self.Fs = forward_propagation(Xs, params, 'feature extractor')
        self.Ft = forward_propagation(Xt, params, 'feature extractor')
        self.F = tf.concat([self.Fs, self.Ft], axis=0)
        self.F_ = _flip_gradient(self.F, learning_rate=l)
    
        out_label = forward_propagation(self.Fs, params, 'label predictor')
        out_domain = forward_propagation(self.F_, params, 'domain classifier')

        self.L_label = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_label, labels=Ys))
        self.L_domain = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_domain, labels=D))
    
        self.L_final = tf.add(self.L_label, self.L_domain)

        self.label_train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.L_label)
        self.domain_train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.L_domain)
        self.final_train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.L_final)

        y_hat = tf.argmax(tf.nn.softmax(out_label), 1)
        d_hat = tf.argmax(tf.nn.softmax(out_domain), 1)

        self.acc_label = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Ys, 1), y_hat), tf.float32))
        self.acc_domain = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(D, 1), d_hat), tf.float32))

    def train(self, data_S, data_T, trainingMode, num_epochs=100, minibatch_size=500):
        seed = 9449
    
        if trainingMode == 'dann':
            minibatch_size = minibatch_size//2

        sess = tf.compat.v1.InteractiveSession()

        # Run the initialization
        sess.run(tf.compat.v1.global_variables_initializer())
            
        # Do the training loop
        for epoch in range(num_epochs):
            loss_f = 0.; loss_d = 0.; loss_p = 0.
            
            # number of minibatches of size minibatch_size in the train set
            # for the convenience, the size of the learning set and the verification set are the same.
            num_batches = int(data_S['x_train'].shape[0] / minibatch_size)
            seed = (seed + 2229)%3114
            
            batches_S = random_mini_batches(data_S['x_train'], data_S['y_train'],
                                            mini_batch_size=minibatch_size, seed=seed)
            batches_T = random_mini_batches(data_T['x_train'], data_T['y_train'],
                                            mini_batch_size=minibatch_size, seed=seed)
            
            for i, minibatch_S, minibatch_T in zip(range(num_batches), batches_S, batches_T):
                # Select a minibatch
                (X_s, Y_s) = minibatch_S
                (X_t, Y_t) = minibatch_T
                
                p = float(i) / num_batches
                #l_p = 2. / (1. + np.exp(-10. * p)) - 1
                l_p = 1.
                
                if trainingMode == 'source only' :
                    _, _loss_p = sess.run([self.label_train_op, self.L_label], 
                                          feed_dict={'Xs:0': X_s, 'Ys:0': Y_s})                
                    loss_p = loss_p + _loss_p / num_batches
                elif trainingMode == 'target only' :
                    _, _loss_p = sess.run([self.label_train_op, self.L_label], 
                                          feed_dict={'Xs:0': X_t, 'Ys:0': Y_t})                
                    loss_p = loss_p + _loss_p / num_batches
                elif trainingMode == 'dann' :
                    D_ = np.vstack([np.repeat([[1,0]], minibatch_size, axis=0), 
                                    np.repeat([[0,1]], minibatch_size, axis=0)])
            
                    _, _loss_f, _loss_d, _loss_p = sess.run([self.final_train_op, self.L_final, self.L_domain, self.L_label],
                                                        feed_dict={'Xs:0': X_s, 'Ys:0': Y_s, 'Xt:0': X_t, 'Yt:0': Y_t,
                                                                   'D:0': D_, 'l:0': l_p})
                    loss_f = loss_f + _loss_f / num_batches
                    loss_d = loss_d + _loss_d / num_batches
                    loss_p = loss_p + _loss_p / num_batches
                else :
                    print("You have passed a wrong argument\n",
                        "Training method should be one among 'source only', 'target only', 'dann'")
                    exit()
                    
            acc_ps, acc_ds = sess.run([self.acc_label, self.acc_domain],
                                feed_dict={'Xs:0': data_S['x_val'], 'Ys:0': data_S['y_val'],
                                           'Xt:0': np.zeros(shape=(0,28,28,3), dtype=np.float32), 
                                           'Yt:0': np.zeros(shape=(0,10), dtype=np.float32),
                                           'D:0': np.repeat([[1,0]], data_S['x_val'].shape[0], axis=0)})
            acc_pt, acc_dt = sess.run([self.acc_label, self.acc_domain],
                                feed_dict={'Xs:0': data_T['x_val'], 'Ys:0': data_T['y_val'],
                                           'Xt:0': np.zeros(shape=(0,28,28,3), dtype=np.float32), 
                                           'Yt:0': np.zeros(shape=(0,10), dtype=np.float32),
                                           'D:0': np.repeat([[0,1]], data_T['x_val'].shape[0], axis=0)})
                    
            report = "Epoch %i:\n  └LOSS: "%(epoch+1)
            
            if trainingMode == 'dann':
                report += "total (%.3f), domain (%.3f), label (%.3f)\n  └ACC: "%(loss_f, loss_d, loss_p)
            else :
                report += "label (%.3f)\n  └ACC: "%(loss_p)
                
            report += "label (S : %.3f/ T : %.3f), domain (S : %.3f/ T : %.3f)"%(acc_ps, acc_pt, acc_ds, acc_dt)
                
            print(report)
            
        samples = np.random.choice(range(data_S['x_val'].shape[0]), 250, replace=False)
        imgs = np.vstack([data_S['x_val'][samples], data_T['x_val'][samples]])
        labels = [0]*250 + [1]*250
        
        f = sess.run(self.Fs, feed_dict={'Xs:0': imgs})
        visualize_tsne(f, labels)

        sess.close()

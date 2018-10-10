#! /usr/bin/python

from model import *
import pickle

# Utils ----------------------------------------------------------------------
def create_random_mat(shape, listidx=None):
    """
    This function create a random sparse index matrix with a given shape. It
    is useful to create negative triplets.

    :param shape: shape of the desired sparse matrix.
    :param listidx: list of index to sample from (default None: it samples from
                    all shape[0] indexes).

    :note: if shape[1] > shape[0], it loops over the shape[0] indexes.
    """
    if listidx is None:
        listidx = np.arange(shape[0])
    listidx = listidx[np.random.permutation(len(listidx))]
    cooData = np.ones(shape[1], dtype=theano.config.floatX)
    cooRowIdxs = listidx[np.arange(shape[1]) % len(listidx)]
    cooColIdxs = range(shape[1])
    randommat = scipy.sparse.coo_matrix((cooData, (cooRowIdxs, cooColIdxs)), shape=shape)
    return scipy.sparse.csc_matrix(randommat)


def load_file(path):
    return scipy.sparse.csr_matrix(pickle.load(open(path, "rb")),
            dtype=theano.config.floatX)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]


class DD(dict):
    """This class is only used to replace a state variable of Jobman"""

    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.iteritems():
            z[k] = copy.deepcopy(kv, memo)
        return z

# ----------------------------------------------------------------------------


# Experiment function --------------------------------------------------------
def WNexp(state, channel):

    # Show experiment parameters
    print(sys.stderr, state)
    np.random.seed(state.seed)

    # Experiment folder
    if hasattr(channel, 'remote_path'):
        state.savepath = channel.remote_path + '/'
    elif hasattr(channel, 'path'):
        state.savepath = channel.path + '/'
    else:
        if not os.path.isdir(state.savepath):
            os.mkdir(state.savepath)
    # if not (state.incre or state.decre):
    #     dataset = state.dataset
    # elif state.incre:
    #     dataset = 'incre'
    # elif state.decre:
    #     dataset = 'decre'
    dataset = state.dataset
    # Positives

    if state.decre:
        trainl = load_file(state.decre + dataset + '-train-lhs.pkl')
        trainr = load_file(state.decre + dataset + '-train-rhs.pkl')
        traino = load_file(state.decre + dataset + '-train-rel.pkl')
        # trainl = load_file(state.decre + 'decre-lhs-lhs.pkl')
        # trainr = load_file(state.decre + 'decre-lhs-rhs.pkl')
        # traino = load_file(state.decre + 'decre-lhs-rel.pkl')
        if state.op == 'SE' or state.op == 'TransE':
            traino = traino[-state.Nrel:, :]

        # Valid set
        validl = load_file(state.decre + dataset + '-valid-lhs.pkl')
        validr = load_file(state.decre + dataset + '-valid-rhs.pkl')
        valido = load_file(state.decre + dataset + '-valid-rel.pkl')
        # validl = load_file(state.decre + 'WN-valid-lhs.pkl')
        # validr = load_file(state.decre + 'WN-valid-rhs.pkl')
        # valido = load_file(state.decre + 'WN-valid-rel.pkl')
        if state.op == 'SE' or state.op == 'TransE':
            valido = valido[-state.Nrel:, :]

        # Test set
        testl = load_file(state.decre + dataset + '-test-lhs.pkl')
        testr = load_file(state.decre + dataset + '-test-rhs.pkl')
        testo = load_file(state.decre + dataset + '-test-rel.pkl')
        # testl = load_file(state.decre + 'WN-test-lhs.pkl')
        # testr = load_file(state.decre + 'WN-test-rhs.pkl')
        # testo = load_file(state.decre + 'WN-test-rel.pkl')
        if state.op == 'SE' or state.op == 'TransE':
            testo = testo[-state.Nrel:, :]
    elif state.incre:
        trainl = load_file(state.incre + dataset + '-train-lhs.pkl')
        trainr = load_file(state.incre + dataset + '-train-rhs.pkl')
        traino = load_file(state.incre + dataset + '-train-rel.pkl')
        # trainl = load_file(state.incre + 'incre-train-lhs.pkl')
        # trainr = load_file(state.incre + 'incre-train-rhs.pkl')
        # traino = load_file(state.incre + 'incre-train-rel.pkl')
        # if state.op == 'SE' or state.op == 'TransE':
        #     traino = traino[-state.Nrel:, :]
        # Valid set
        validl = load_file(state.incre + dataset + '-valid-lhs.pkl')
        validr = load_file(state.incre + dataset + '-valid-rhs.pkl')
        valido = load_file(state.incre + dataset + '-valid-rel.pkl')
        if state.op == 'SE' or state.op == 'TransE':
            valido = valido[-state.Nrel:, :]

        # Test set
        testl = load_file(state.incre + dataset + '-test-lhs.pkl')
        testr = load_file(state.incre + dataset + '-test-rhs.pkl')
        testo = load_file(state.incre + dataset + '-test-rel.pkl')
        if state.op == 'SE' or state.op == 'TransE':
            testo = testo[-state.Nrel:, :]
    else:
        trainl = load_file(state.datapath + dataset + '-train-lhs.pkl')
        trainr = load_file(state.datapath + dataset + '-train-rhs.pkl')
        traino = load_file(state.datapath + dataset + '-train-rel.pkl')
        if state.op == 'SE' or state.op == 'TransE':
            traino = traino[-state.Nrel:, :]

        # Valid set
        validl = load_file(state.datapath + 'WN-valid-lhs.pkl')
        validr = load_file(state.datapath + 'WN-valid-rhs.pkl')
        valido = load_file(state.datapath + 'WN-valid-rel.pkl')
        if state.op == 'SE' or state.op == 'TransE':
            valido = valido[-state.Nrel:, :]

        # Test set
        testl = load_file(state.datapath + 'WN-test-lhs.pkl')
        testr = load_file(state.datapath + 'WN-test-rhs.pkl')
        testo = load_file(state.datapath + 'WN-test-rel.pkl')
        if state.op == 'SE' or state.op == 'TransE':
            testo = testo[-state.Nrel:, :]

    # Index conversion
    trainlidx = convert2idx(trainl)[:state.neval]
    trainridx = convert2idx(trainr)[:state.neval]
    trainoidx = convert2idx(traino)[:state.neval]
    validlidx = convert2idx(validl)[:state.neval]
    validridx = convert2idx(validr)[:state.neval]
    validoidx = convert2idx(valido)[:state.neval]
    testlidx = convert2idx(testl)[:state.neval]
    testridx = convert2idx(testr)[:state.neval]
    testoidx = convert2idx(testo)[:state.neval]

    # Model declaration
    if not state.loadmodel:
        # operators
        if state.op == 'Unstructured':
            leftop = Unstructured()
            rightop = Unstructured()
        elif state.op == 'SME_lin':
            leftop = LayerLinear(np.random, 'lin', state.ndim, state.ndim,
                    state.nhid, 'left')
            rightop = LayerLinear(np.random, 'lin', state.ndim, state.ndim,
                    state.nhid, 'right')
        elif state.op == 'SME_bil':
            leftop = LayerBilinear(np.random, 'lin', state.ndim, state.ndim,
                    state.nhid, 'left')
            rightop = LayerBilinear(np.random, 'lin', state.ndim, state.ndim,
                    state.nhid, 'right')
        elif state.op == 'SE':
            leftop = LayerMat('lin', state.ndim, state.nhid)
            rightop = LayerMat('lin', state.ndim, state.nhid)
        elif state.op == 'TransE':
            leftop  = LayerTrans()
            rightop = Unstructured()
        # embeddings
        if not state.loademb:
            embeddings = Embeddings(np.random, state.Nent, state.ndim, 'emb')
        else:
            f = open(state.loademb, "rb")
            embeddings = pickle.load(f, encoding='iso-8859-1')
            f.close()
        if state.op == 'SE' and type(embeddings) is not list:
            relationl = Embeddings(np.random, state.Nrel,
                    state.ndim * state.nhid, 'rell')
            relationr = Embeddings(np.random, state.Nrel,
                    state.ndim * state.nhid, 'relr')
            embeddings = [embeddings, relationl, relationr]
        if state.op == 'TransE' and type(embeddings) is not list:
            relationVec = Embeddings(np.random, state.Nrel, state.ndim, 'relvec')
            embeddings = [embeddings, relationVec, relationVec]
        simfn = eval(state.simfn + 'sim')
        trainfunc = TrainFn1Member(simfn, embeddings, leftop, rightop,
                                   marge=state.marge, rel=False)
    else:
        f = open(state.loadmodel, "rb")
        embeddings = pickle.load(f, encoding='iso-8859-1')
        leftop = pickle.load(f, encoding='iso-8859-1')
        rightop = pickle.load(f, encoding='iso-8859-1')
        simfn = pickle.load(f, encoding='iso-8859-1')
        f.close()
        if not (state.incre or state.decre):
            trainfunc = TrainFn1Member(simfn, embeddings, leftop, rightop,
                                       marge=state.marge, rel=False)
        elif state.incre:
            D = state.ndim
            wbound = np.sqrt(6. / D)
            f_incre = open(state.incre + 'incre_entity_list.pkl', 'rb')
            num_list = pickle.load(f_incre, encoding='iso-8859-1')
            f_incre.close()
            x = embeddings.E
            y = x.get_value().tolist()
            for num in num_list:
                for i in range(0, 100):
                    W_values = np.random.uniform(low=-wbound, high=wbound)
                    y[i].insert(num, W_values)
            y = np.array(y)
            embeddings.E.set_value(y)
            f1 = open(state.incre + 'incre_best_valid_model.pkl', 'wb')
            pickle.dump(embeddings, f1, -1)
            pickle.dump(leftop, f1, -1)
            pickle.dump(rightop, f1, -1)
            pickle.dump(simfn, f1, -1)
            f1.close()
            trainfunc = TrainFn1Member(simfn, embeddings, leftop, rightop,
                                       marge=state.marge, rel=False)
        elif state.decre:

            ######################################################
            f_decre = open(state.decre + 'decre_num_list.pkl', 'rb')
            num_list = pickle.load(f_decre, encoding='iso-8859-1')
            f_decre.close()
            num_list = sorted(num_list, reverse=True)
            x = embeddings.E
            y = x.get_value().tolist()
            for num in num_list:
                # print num
                for i in range(0, 100):
                    del y[i][num]
            y = np.array(y)
            # print y.shape
            embeddings.E.set_value(y)
            f = open(state.decre + 'decre_best_valid_model.pkl', 'wb')
            pickle.dump(embeddings, f, -1)
            pickle.dump(leftop, f, -1)
            pickle.dump(rightop, f, -1)
            pickle.dump(simfn, f, -1)
            f.close()
            trainfunc = TrainFn1Member(simfn, embeddings, leftop, rightop,
                                       marge=state.marge, rel=False)

            ######################################################
            # if state.postion == 'lhs':
            #     trainfunc = DecreTrainFn1Member(simfn, embeddings, leftop, rightop,
            #                            marge=state.marge, rel=False, postion=state.postion)
            # elif state.postion == 'rhs':
            #     trainfunc = DecreTrainFn1Member(simfn, embeddings, leftop, rightop,
            #                                 marge=state.marge, rel=False, postion=state.postion)


    # Function compilation

    ranklfunc = RankLeftFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=state.Nsyn)
    rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=state.Nsyn)

    out = []
    outb = []
    state.bestvalid = -1

    batchsize = int(trainl.shape[1] / state.nbatches)

    print(sys.stderr, "BEGIN TRAINING")
    timeref = time.time()
    totepochs = state.totepochs
    test_all = state.test_all
    if state.decre:
        totepochs = 100
        test_all = 100
    for epoch_count in range(1, totepochs + 1):
        # Shuffling
        order = np.random.permutation(trainl.shape[1])
        trainl = trainl[:, order]
        trainr = trainr[:, order]
        traino = traino[:, order]
        # Negatives
        trainln = create_random_mat(trainl.shape, np.arange(state.Nsyn))
        trainrn = create_random_mat(trainr.shape, np.arange(state.Nsyn))

        for i in range(state.nbatches):
            tmpl = trainl[:, range(i * batchsize, (i + 1) * batchsize)]
            tmpr = trainr[:, range(i * batchsize, (i + 1) * batchsize)]
            tmpo = traino[:, range(i * batchsize, (i + 1) * batchsize)]
            tmpnl = trainln[:, range(i * batchsize, (i + 1) * batchsize)]
            tmpnr = trainrn[:, range(i * batchsize, (i + 1) * batchsize)]
            # training iteration
            outtmp = trainfunc(state.lremb, state.lrparam / float(batchsize),
                    tmpl, tmpr, tmpo, tmpnl, tmpnr)
            out += [outtmp[0] / float(batchsize)]
            outb += [outtmp[1]]
            # embeddings normalization
            if type(embeddings) is list:
                embeddings[0].normalize()
            else:
                embeddings.normalize()

        if (epoch_count % test_all) == 0:
            # model evaluation
            print(sys.stderr, "-- EPOCH %s (%s seconds per epoch):" % (
                    epoch_count,
                    round(time.time() - timeref, 3) / float(test_all)))
            timeref = time.time()
            print(sys.stderr, "COST >> %s +/- %s, %% updates: %s%%" % (
                    round(np.mean(out), 4), round(np.std(out), 4),
                    round(np.mean(outb) * 100, 3)))
            out = []
            outb = []
            resvalid = RankingScoreIdx(ranklfunc, rankrfunc,
                    validlidx, validridx, validoidx)
            state.valid = np.mean(resvalid[0] + resvalid[1])
            restrain = RankingScoreIdx(ranklfunc, rankrfunc,
                    trainlidx, trainridx, trainoidx)
            state.train = np.mean(restrain[0] + restrain[1])
            print(sys.stderr, "\tMEAN RANK >> valid: %s, train: %s" % (
                    state.valid, state.train))
            if state.bestvalid == -1 or state.valid < state.bestvalid:
                restest = RankingScoreIdx(ranklfunc, rankrfunc,
                        testlidx, testridx, testoidx)
                state.bestvalid = state.valid
                state.besttrain = state.train
                state.besttest = np.mean(restest[0] + restest[1])
                state.bestepoch = epoch_count
                # Save model best valid model
                if not (state.incre or state.decre):
                    f = open(state.savepath + 'best_valid_model.pkl', 'wb')
                    pickle.dump(embeddings, f, -1)
                    pickle.dump(leftop, f, -1)
                    pickle.dump(rightop, f, -1)
                    pickle.dump(simfn, f, -1)
                    f.close()
                elif state.incre:
                    f = open(state.incre + 'best_valid_model.pkl', 'wb')
                    pickle.dump(embeddings, f, -1)
                    pickle.dump(leftop, f, -1)
                    pickle.dump(rightop, f, -1)
                    pickle.dump(simfn, f, -1)
                    f.close()
                elif state.decre:
                    f = open(state.decre + 'best_valid_model.pkl', 'wb')
                    pickle.dump(embeddings, f, -1)
                    pickle.dump(leftop, f, -1)
                    pickle.dump(rightop, f, -1)
                    pickle.dump(simfn, f, -1)
                    f.close()
                    ######################################################
                    # # print '\t##### REMOVE ENTITIES#####'
                    # f_decre = open(state.decre + 'decre_num_list.pkl', 'r')
                    # num_list = pickle.load(f_decre)
                    # f_decre.close()
                    # num_list = sorted(num_list, reverse=True)
                    # x = embeddings.E
                    # y = x.get_value().tolist()
                    # for num in num_list:
                    #     # print num
                    #     for i in range(0, 100):
                    #         del y[i][num]
                    # y = np.array(y)
                    # # print y.shape
                    # embeddings.E.set_value(y)
                    # f = open(state.decre + 'decre_best_valid_model.pkl', 'wb')
                    # pickle.dump(embeddings, f, -1)
                    # pickle.dump(leftop, f, -1)
                    # pickle.dump(rightop, f, -1)
                    # pickle.dump(simfn, f, -1)
                    # f.close()
            ######################################################

                print(sys.stderr, "\t\t##### NEW BEST VALID >> test: %s" % (
                        state.besttest))
            # Save current model
            f = open(state.savepath + '/current_model.pkl', 'wb')
            pickle.dump(embeddings, f, -1)
            pickle.dump(leftop, f, -1)
            pickle.dump(rightop, f, -1)
            pickle.dump(simfn, f, -1)
            f.close()
            state.nbepochs = epoch_count
            print(sys.stderr, "\t(the evaluation took %s seconds)" % (
                round(time.time() - timeref, 3)))
            timeref = time.time()
            channel.save()


    if state.incre:
        launch(op='SME_lin', simfn='Dot', ndim=100, nhid=50, marge=1., lremb=0.01,
               lrparam=1.0, nbatches=100, totepochs=500, test_all=500,
               savepath=state.savepath,
               datapath=state.savepath, dataset='incre', Nent=state.Nent,
               Nsyn=state.Nsyn, Nrel=state.Nrel,
               loadmodel=state.savepath + ('best_valid_model.pkl'),
               # loadmodel=False,
               loademb=False,
               # incre=state.incre,
               # decre='/Users/a/Downloads/SME-master2/data/decrement/1w/changed/',
               increent=0, postion='lhs', neval=50, seed=123)
    return channel.COMPLETE




def launch(datapath='', dataset='WN', Nent=1989,
        Nsyn=1984, Nrel=5, loadmodel=False,
           loademb=True, incre=False,
           decre=False, postion = '', op='Unstructured',
        simfn='Dot', ndim=50, nhid=50, marge=1., lremb=0.1, lrparam=1., increent = 0,
        nbatches=100, totepochs=2000, test_all=1, neval=50, seed=123,
        savepath=''):

    # Argument of the experiment script
    state = DD()

    state.datapath = datapath
    state.dataset = dataset
    state.Nent = Nent
    state.Nsyn = Nsyn
    state.Nrel = Nrel
    state.loadmodel = loadmodel
    state.loademb = loademb
    state.incre = incre
    state.decre = decre
    state.op = op
    state.increent = increent
    state.simfn = simfn
    state.ndim = ndim
    state.nhid = nhid
    state.marge = marge
    state.lremb = lremb
    state.lrparam = lrparam
    state.nbatches = nbatches
    state.totepochs = totepochs
    state.test_all = test_all
    state.neval = neval
    state.seed = seed
    state.savepath = savepath
    state.postion = postion

    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    # Jobman channel remplacement
    class Channel(object):
        def __init__(self, state):
            self.state = state
            f = open(self.state.savepath + '/orig_state.pkl', 'wb')
            pickle.dump(self.state, f, -1)
            f.close()
            self.COMPLETE = 1

        def save(self):
            f = open(self.state.savepath + '/current_state.pkl', 'wb')
            pickle.dump(self.state, f, -1)
            f.close()

    channel = Channel(state)

    WNexp(state, channel)

if __name__ == '__main__':
    launch()

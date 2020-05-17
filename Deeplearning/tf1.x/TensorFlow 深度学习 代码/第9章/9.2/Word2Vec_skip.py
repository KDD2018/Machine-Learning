import tensorflow as tf
import numpy as np
import math
import vocabulary

max_steps = 100000  # 训练最大迭代次数10w次
batch_size = 128
embedding_size = 128  # 嵌入向量的尺寸
skip_distance = 1  # 相邻单词数
num_of_samples = 2  # 对每个单词生成多少样本

vocabulary_size = 50000  # 词汇量

# numpy中choice()函数的函数原型为choice(a,size,replace,p)
# choice()函数用于在a给出的范围内抽取size个大小的数组成一个一维数组
# 当设置了replace=False则表示组成的这个一维数组中不能够有重复的数字
valid_examples = np.random.choice(100, 16, replace=False)

num_sampled = 64  # 训练时用来做负样本的噪声单词的数量

with tf.Graph().as_default():
    # train_inputs和train_labels是训练数据及其label的placeholder
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # embeddings是所有50000高频单词的词向量，向量的维度是128，数值是由
    # random_uniform()函数生成的在-1.0到1.0之间平均分布的数值
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    # embedding_lookup()函数用于选取一个张量里面索引对应的元素，函数原型是：
    # embedding_lookup(params,ids,partition_strategy,name,validate_indices,max_norm)
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 用truncated_normal()函数产生标准差为1.0/math.sqrt(embedding_size)的正态分布数据
    # 产生的nce_weights作为NCE loss中的权重参数
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=1.0 / math.sqrt(embedding_size)))

    # 产生的nce_biases作为NCE loss中的偏置参数
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # 计算词向量embedding在训练数据上的loss
    # nce_loss()函数原型nce_loss(weights, biases, inputs, labels, num_sampled, num_classes,
    #                                    num_true=1,sampled_values,remove_accidental_hits,
    #                                                             partition_strategy,name)
    nec_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                              labels=train_labels, inputs=embed,
                              num_sampled=num_sampled,
                              num_classes=vocabulary_size)

    # 求nce_loss的均值
    loss = tf.reduce_mean(nec_loss)

    # 创建优化器，学习率为固定的1.0，最小化loss
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # square()函数用于求平方，之后使用reduce_sum()函数求和
    # keep_dims=True表示求和之后维度不会发生改变
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

    normalized_embeddings = embeddings / norm

    # 在标准化后的所有单词的词向量值中寻找随机抽取的16个单词对应的词向量值
    # 在这之前，valid_inputs是由数组valid_examples进行constant操作转化为张量得来，
    valid_inputs = tf.constant(valid_examples, dtype=tf.int32)
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_inputs)

    # 使用matmul()函数计算相似度
    # 函数原型matmul(a, b, transpose_a, transpose_b, a_is_sparse, b_is_sparse, name)
    # 在函数matmul()的定义中，name参数默认为None，除a和b外其他参数都有默认的
    # False值，在这里我们设参数transpose_b设True，表示对参数b传入的矩阵进行转置
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # 开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 总损失与平均损失
        total_loss = 0
        average_loss = 0

        for step in range(max_steps + 1):

            # 调用generate_batch()函数生成用于训练的batch及其labels，
            batch_inputs, batch_labels = vocabulary.generate_batch(
                batch_size, num_of_samples, skip_distance)

            # 运行loss的计算及最小化loss的优化器
            loss_val, _ = sess.run([loss, optimizer], feed_dict={train_inputs: batch_inputs,
                                                                 train_labels: batch_labels})

            # total_loss用于计算总损失，在每一轮迭代后都会与loos_val相加
            total_loss += loss_val

            # 每进行1000轮迭代就输出平均损失的值，并将average_loss和total_loss
            # 重新归零，方便下一个1000轮的计算
            if step > 0 and step % 1000 == 0:
                average_loss = total_loss / 1000
                print("Average loss at %d step is:%f " % (step, average_loss))
                average_loss = 0
                total_loss = 0

            # 每隔5000轮就打印一次与验证单词最相似的8个单词
            if step > 0 and step % 5000 == 0:

                # 执行计算相似性的操作
                similar = similarity.eval()

                # 外层循环16次，
                for i in range(16):

                    # 每执行一次最外层的循环，都会得到一个验证单词对应的nearest，
                    # 这里有8个数据，是与验证单词最相近的单词的编号，通过
                    # reverse_dictionary可以得到确切的单词
                    nearest = (-similar[i, :]).argsort()[1:8 + 1]

                    # 定义需要打印的字符串，其中valid_word是通过reverse_dictionary得到的验证单词
                    valid_word = vocabulary.reverse_dictionary[valid_examples[i]]
                    nearest_information = "Nearest to %s is:" % valid_word

                    for j in range(8):
                        # 在8个循环内通过reverse_dictionary得到与验证单词相近的8个单词的原型
                        # 并改进需要打印的字符串
                        close_word = vocabulary.reverse_dictionary[nearest[j]]
                        nearest_information = " %s %s" % (nearest_information,close_word)

                    # 打印出验证单词及与验证单词相近的8个单词
                    print("valid_word is: %s"% valid_word)
                    print(nearest_information)

        final_embeddings = normalized_embeddings.eval()

'''打印的信息
len of original word is 17005207
Average loss at 1000 step is:145.009534 
Average loss at 2000 step is:84.632241 
Average loss at 3000 step is:58.798290 
Average loss at 4000 step is:46.023421 
Average loss at 5000 step is:35.622134 
valid_word is: state
        Nearest to state is: we noted reginae ants atlantic facts part asparagales
valid_word is: states
        Nearest to states is: cl acres aircraft victoriae expansion victim investment fullback
valid_word is: it
        Nearest to it is: preserve turned victoriae collation everyone gland opium lining
valid_word is: up
        Nearest to up is: columbia austin segments gland reginae metaphysics faith once
valid_word is: between
        Nearest to between is: populated mourned grow reginae deprived play ada afl
valid_word is: with
        Nearest to with is: and ada if programming acting aq carry suggested
valid_word is: use
        Nearest to use is: focuses austin reginae victoriae broadcasts influence out wire
valid_word is: his
        Nearest to his is: victoriae alignment perform cl byrd dynamics kick perennials
valid_word is: two
        Nearest to two is: victoriae cambodia one austin agave nine mouth outline
valid_word is: three
        Nearest to three is: cl agave zero victoriae reginae poets austin archaeologists
valid_word is: but
        Nearest to but is: built darius conference ayn geology cm victoriae gland
valid_word is: more
        Nearest to more is: cl victoriae soap until york syllabary august agave
valid_word is: the
        Nearest to the is: a and victoriae one ptolemy unkown austin of
valid_word is: in
        Nearest to in is: and of to on emmy petroleum victoriae initial
valid_word is: united
        Nearest to united is: functions und reginae loyalists linebackers roman campaign austin
valid_word is: during
        Nearest to during is: results nozick cambodia gland linguistics victoriae captured russia
Average loss at 6000 step is:30.947824 
Average loss at 7000 step is:24.874575 
Average loss at 8000 step is:22.010970 
Average loss at 9000 step is:19.140614 
Average loss at 10000 step is:16.452675 
valid_word is: state
        Nearest to state is: casement noted we ants flee atlantic reginae however
valid_word is: states
        Nearest to states is: cl aircraft expansion acres victoriae victim kept architectural
valid_word is: it
        Nearest to it is: he victoriae preserve hustler bang drawings gland cl
valid_word is: up
        Nearest to up is: granada sarcoma columbia segments enlightened fervent austin metaphysics
valid_word is: between
        Nearest to between is: deprived mourned populated one grow in afl play
valid_word is: with
        Nearest to with is: and in of easy if motorcycle markov from
valid_word is: use
        Nearest to use is: focuses austin broadcasts reginae victoriae markov amo influence
valid_word is: his
        Nearest to his is: the their victoriae a alignment longest perform dose
valid_word is: two
        Nearest to two is: one victoriae unkown cambodia nine austin reginae markov
valid_word is: three
        Nearest to three is: nine victoriae zero four reginae agave one five
valid_word is: but
        Nearest to but is: built and conference ayn darius scenario geology posters
valid_word is: more
        Nearest to more is: cl roper modes victoriae soap nextstep lords until
valid_word is: the
        Nearest to the is: a austin unkown his victoriae hydrate one ptolemy
valid_word is: in
        Nearest to in is: of and on at with reginae by unkown
valid_word is: united
        Nearest to united is: pedro functions und car amo conversely reginae perimeter
valid_word is: during
        Nearest to during is: results nozick architect at receivers cambodia stands kicking
Average loss at 11000 step is:14.630424 
Average loss at 12000 step is:13.392861 
Average loss at 13000 step is:12.213534 
Average loss at 14000 step is:10.764199 
Average loss at 15000 step is:10.540506 
valid_word is: state
        Nearest to state is: meridians flee noted ants casement tats we atlantic
valid_word is: states
        Nearest to states is: pains expansion acres cl hello siena aircraft victoriae
valid_word is: it
        Nearest to it is: he victoriae they bang cl hustler this preserve
valid_word is: up
        Nearest to up is: granada sarcoma columbia dasyprocta lignite fervent enlightened trumps
valid_word is: between
        Nearest to between is: deprived in mourned grow populated at meridians practised
valid_word is: with
        Nearest to with is: in and from dasyprocta of easy to as
valid_word is: use
        Nearest to use is: agouti broadcasts focuses dasyprocta austin amo markov convicted
valid_word is: his
        Nearest to his is: the their its a victoriae s longest dose
valid_word is: two
        Nearest to two is: one nine three eight five four zero dasyprocta
valid_word is: three
        Nearest to three is: two zero nine four five eight one six
valid_word is: but
        Nearest to but is: agouti and dasyprocta built epistle geology meridians ayn
valid_word is: more
        Nearest to more is: roper cl modes quarried lords aim dasyprocta drivers
valid_word is: the
        Nearest to the is: a his and dasyprocta austin its agouti some
valid_word is: in
        Nearest to in is: of and at on with by for nine
valid_word is: united
        Nearest to united is: pedro functions und octavia weaver clement oblique calvinism
valid_word is: during
        Nearest to during is: results nozick at architect receivers kicking expounded stands
Average loss at 16000 step is:9.183173 
Average loss at 17000 step is:8.530489 
Average loss at 18000 step is:8.715831 
Average loss at 19000 step is:8.228534 
Average loss at 20000 step is:7.280190 
valid_word is: state
        Nearest to state is: tats flee ants meridians noted casement however atlantic
valid_word is: states
        Nearest to states is: imran brine pains stabbed acres cl hello siena
valid_word is: it
        Nearest to it is: he they this victoriae antipope bang cl not
valid_word is: up
        Nearest to up is: granada sarcoma columbia dasyprocta trumps lignite atop segments
valid_word is: between
        Nearest to between is: in deprived mourned and by of at populated
valid_word is: with
        Nearest to with is: in and from of by for dasyprocta as
valid_word is: use
        Nearest to use is: agouti broadcasts dasyprocta focuses austin convicted out influence
valid_word is: his
        Nearest to his is: the their its s a victoriae longest dose
valid_word is: two
        Nearest to two is: one three five four eight zero nine six
valid_word is: three
        Nearest to three is: two four zero eight five six nine dasyprocta
valid_word is: but
        Nearest to but is: agouti and dasyprocta built meridians victoriae that gland
valid_word is: more
        Nearest to more is: most quarried roper three modes cl lords aim
valid_word is: the
        Nearest to the is: a his its dasyprocta agouti victoriae their some
valid_word is: in
        Nearest to in is: at with of and on from by for
valid_word is: united
        Nearest to united is: pedro functions octavia und weaver clement nlm conversely
valid_word is: during
        Nearest to during is: at receivers for results architect in nozick clergyman
Average loss at 21000 step is:7.174989 
Average loss at 22000 step is:7.416212 
Average loss at 23000 step is:6.940413 
Average loss at 24000 step is:6.959669 
Average loss at 25000 step is:6.733032 
valid_word is: state
        Nearest to state is: ramps meridians flee tats noted ants casement predates
valid_word is: states
        Nearest to states is: stabbed pains acres imran brine siena aircraft expansion
valid_word is: it
        Nearest to it is: he this they victoriae there not antipope aediles
valid_word is: up
        Nearest to up is: granada sarcoma columbia trumps dasyprocta atop integrity lignite
valid_word is: between
        Nearest to between is: in deprived mourned at by afl populated five
valid_word is: with
        Nearest to with is: in and from by for or as dasyprocta
valid_word is: use
        Nearest to use is: agouti focuses broadcasts dasyprocta dek convicted austin out
valid_word is: his
        Nearest to his is: their the its s her dose alignment a
valid_word is: two
        Nearest to two is: one three four five six eight nine agouti
valid_word is: three
        Nearest to three is: eight four six five two nine seven zero
valid_word is: but
        Nearest to but is: agouti and dasyprocta built meridians gland victoriae reginae
valid_word is: more
        Nearest to more is: most three quarried modes roper cl hamburger ramps
valid_word is: the
        Nearest to the is: a his their dasyprocta its agouti victoriae ramps
valid_word is: in
        Nearest to in is: at from on nine and of with initial
valid_word is: united
        Nearest to united is: pedro octavia of und functions perimeter weaver clement
valid_word is: during
        Nearest to during is: at for in receivers clergyman results architect chesapeake
Average loss at 26000 step is:6.602586 
Average loss at 27000 step is:6.366769 
Average loss at 28000 step is:5.879195 
Average loss at 29000 step is:6.114192 
Average loss at 30000 step is:6.131620 
valid_word is: state
        Nearest to state is: ramps meridians ants tats flee casement noted stadium
valid_word is: states
        Nearest to states is: pains stabbed acres brine imran siena cl aircraft
valid_word is: it
        Nearest to it is: he this they there akita not victoriae antipope
valid_word is: up
        Nearest to up is: granada trumps sarcoma columbia dasyprocta atop segments abkhazians
valid_word is: between
        Nearest to between is: in deprived mourned by davids at practised of
valid_word is: with
        Nearest to with is: in from by for and as or dasyprocta
valid_word is: use
        Nearest to use is: agouti broadcasts focuses dek dasyprocta convicted out austin
valid_word is: his
        Nearest to his is: their the its s a her alignment victoriae
valid_word is: two
        Nearest to two is: four three one five six eight seven zero
valid_word is: three
        Nearest to three is: four eight six five two seven nine zero
valid_word is: but
        Nearest to but is: agouti and dasyprocta that meridians victoriae built gland
valid_word is: more
        Nearest to more is: most quarried modes hamburger cl roper ramps lords
valid_word is: the
        Nearest to the is: its a his their akita ramps dasyprocta this
valid_word is: in
        Nearest to in is: at on of with and from nine by
valid_word is: united
        Nearest to united is: pedro primigenius octavia lemonade of weaver und functions
valid_word is: during
        Nearest to during is: in at for by receivers clergyman architect sdi
Average loss at 31000 step is:5.967377 
Average loss at 32000 step is:5.830637 
Average loss at 33000 step is:5.924866 
Average loss at 34000 step is:5.862029 
Average loss at 35000 step is:5.765287 
valid_word is: state
        Nearest to state is: ramps tats ants predates casement meridians flee stadium
valid_word is: states
        Nearest to states is: pains stabbed acres aircraft siena zubaydah brine imran
valid_word is: it
        Nearest to it is: he this there they akita not victoriae gland
valid_word is: up
        Nearest to up is: granada trumps columbia sarcoma dasyprocta once atop segments
valid_word is: between
        Nearest to between is: in deprived mourned by of at davids with
valid_word is: with
        Nearest to with is: in and from by for or as dasyprocta
valid_word is: use
        Nearest to use is: agouti broadcasts focuses dasyprocta out dek convicted austin
valid_word is: his
        Nearest to his is: their its the her s alignment dose a
valid_word is: two
        Nearest to two is: four three one five six seven eight zero
valid_word is: three
        Nearest to three is: four five six eight two seven nine zero
valid_word is: but
        Nearest to but is: agouti and dasyprocta that victoriae meridians gland built
valid_word is: more
        Nearest to more is: most quarried modes hamburger cl three roper ramps
valid_word is: the
        Nearest to the is: akita its a this dasyprocta their his ramps
valid_word is: in
        Nearest to in is: at and on from with of by nine
valid_word is: united
        Nearest to united is: pedro lemonade perimeter octavia primigenius of weaver clement
valid_word is: during
        Nearest to during is: in at for receivers clergyman by liqueur chesapeake
Average loss at 36000 step is:5.624653 
Average loss at 37000 step is:4.989881 
Average loss at 38000 step is:5.553697 
Average loss at 39000 step is:5.567795 
Average loss at 40000 step is:5.324156 
valid_word is: state
        Nearest to state is: ramps meridians tats ants casement flee predates stadium
valid_word is: states
        Nearest to states is: stabbed acres pains aircraft imran guderian zubaydah cl
valid_word is: it
        Nearest to it is: he this there they akita not victoriae and
valid_word is: up
        Nearest to up is: trumps granada dasyprocta sarcoma columbia extract atop once
valid_word is: between
        Nearest to between is: in deprived mourned by at with davids and
valid_word is: with
        Nearest to with is: in and from or for by dasyprocta as
valid_word is: use
        Nearest to use is: agouti dasyprocta broadcasts focuses convicted dek austin abet
valid_word is: his
        Nearest to his is: their its the her s alignment dose a
valid_word is: two
        Nearest to two is: four three five six one seven eight zero
valid_word is: three
        Nearest to three is: four five six eight seven two zero nine
valid_word is: but
        Nearest to but is: and agouti dasyprocta however that victoriae meridians gland
valid_word is: more
        Nearest to more is: most quarried modes hamburger six cl three sleepy
valid_word is: the
        Nearest to the is: a its his their akita dasyprocta ramps this
valid_word is: in
        Nearest to in is: at on and from with of nine reginae
valid_word is: united
        Nearest to united is: pedro of primigenius lemonade perimeter octavia abitibi clement
valid_word is: during
        Nearest to during is: in at for receivers clergyman by liqueur sdi
Average loss at 41000 step is:5.274549 
Average loss at 42000 step is:5.295857 
Average loss at 43000 step is:5.367129 
Average loss at 44000 step is:5.269009 
Average loss at 45000 step is:5.295506 
valid_word is: state
        Nearest to state is: ramps carbohydrates meridians predates stadium flee casement tats
valid_word is: states
        Nearest to states is: stabbed pains acres aircraft imran siena zubaydah cl
valid_word is: it
        Nearest to it is: he this there they akita she victoriae that
valid_word is: up
        Nearest to up is: trumps granada columbia sarcoma dasyprocta extract atop integrity
valid_word is: between
        Nearest to between is: in deprived with mourned by at davids of
valid_word is: with
        Nearest to with is: in from or and for dasyprocta by markov
valid_word is: use
        Nearest to use is: agouti dasyprocta focuses broadcasts dek out convicted abet
valid_word is: his
        Nearest to his is: their its the her s renminbi alignment prism
valid_word is: two
        Nearest to two is: four three six five one eight seven zero
valid_word is: three
        Nearest to three is: four six five eight two seven nine zero
valid_word is: but
        Nearest to but is: and agouti however dasyprocta that victoriae meridians gland
valid_word is: more
        Nearest to more is: most quarried hamburger modes cl three asatru roper
valid_word is: the
        Nearest to the is: its a their his akita dasyprocta this victoriae
valid_word is: in
        Nearest to in is: at on from with for of and nine
valid_word is: united
        Nearest to united is: pedro perimeter lemonade octavia of functions weaver roman
valid_word is: during
        Nearest to during is: in at for receivers clergyman by chesapeake sdi
Average loss at 46000 step is:5.276804 
Average loss at 47000 step is:4.984429 
Average loss at 48000 step is:5.082731 
Average loss at 49000 step is:5.238467 
Average loss at 50000 step is:5.081029 
valid_word is: state
        Nearest to state is: ramps kapoor carbohydrates predates stadium flee meridians casement
valid_word is: states
        Nearest to states is: stabbed pains acres zubaydah siena imran aircraft brine
valid_word is: it
        Nearest to it is: he this there they she akita which victoriae
valid_word is: up
        Nearest to up is: trumps granada extract dasyprocta sarcoma columbia lignite integrity
valid_word is: between
        Nearest to between is: in deprived mourned with by original at seven
valid_word is: with
        Nearest to with is: in or and from dasyprocta by markov thibetanus
valid_word is: use
        Nearest to use is: agouti dasyprocta out dek focuses broadcasts austin abet
valid_word is: his
        Nearest to his is: their its the her s dose altenberg prism
valid_word is: two
        Nearest to two is: three four one five six seven eight zero
valid_word is: three
        Nearest to three is: four six five two eight seven zero nine
valid_word is: but
        Nearest to but is: and agouti however thibetanus dasyprocta meridians victoriae gland
valid_word is: more
        Nearest to more is: most quarried hamburger clearly asatru cl tied modes
valid_word is: the
        Nearest to the is: its their akita his a this agouti dasyprocta
valid_word is: in
        Nearest to in is: at on and from nine kapoor of reginae
valid_word is: united
        Nearest to united is: pedro perimeter lemonade primigenius octavia functions abitibi roman
valid_word is: during
        Nearest to during is: in at for receivers is clergyman by answered
Average loss at 51000 step is:5.210266 
Average loss at 52000 step is:5.162591 
Average loss at 53000 step is:5.127516 
Average loss at 54000 step is:5.150660 
Average loss at 55000 step is:5.055330 
valid_word is: state
        Nearest to state is: ramps kapoor predates stadium meridians carbohydrates casement michelob
valid_word is: states
        Nearest to states is: stabbed acres pains zubaydah aircraft scholarships siena imran
valid_word is: it
        Nearest to it is: he this there they which she akita victoriae
valid_word is: up
        Nearest to up is: trumps granada dasyprocta tours extract columbia integrity sarcoma
valid_word is: between
        Nearest to between is: in deprived with mourned original by afl davids
valid_word is: with
        Nearest to with is: in or by from and dasyprocta markov for
valid_word is: use
        Nearest to use is: agouti dasyprocta broadcasts out abet abitibi dek focuses
valid_word is: his
        Nearest to his is: their its the her s megabats dose altenberg
valid_word is: two
        Nearest to two is: three four five six one eight seven michelob
valid_word is: three
        Nearest to three is: four five six eight two seven nine zero
valid_word is: but
        Nearest to but is: however and agouti michelob thibetanus dasyprocta meridians victoriae
valid_word is: more
        Nearest to more is: most quarried michelob hamburger terribly very tied cl
valid_word is: the
        Nearest to the is: its microbats their akita his this dasyprocta ramps
valid_word is: in
        Nearest to in is: at and from on kapoor of during with
valid_word is: united
        Nearest to united is: pedro perimeter primigenius functions lemonade octavia abitibi roman
valid_word is: during
        Nearest to during is: in at for clergyman receivers by chesapeake thibetanus
Average loss at 56000 step is:5.067634 
Average loss at 57000 step is:5.058398 
Average loss at 58000 step is:5.156872 
Average loss at 59000 step is:4.971608 
Average loss at 60000 step is:4.902193 
valid_word is: state
        Nearest to state is: ramps kapoor meridians michelob predates arms synod casement
valid_word is: states
        Nearest to states is: stabbed acres pains zubaydah imran michelob cl scholarships
valid_word is: it
        Nearest to it is: he this there she they akita callithrix which
valid_word is: up
        Nearest to up is: trumps granada tours extract dasyprocta integrity columbia sarcoma
valid_word is: between
        Nearest to between is: in with deprived mourned holographic original davids by
valid_word is: with
        Nearest to with is: in or fets by and dasyprocta thibetanus markov
valid_word is: use
        Nearest to use is: agouti dasyprocta cebus microcebus dek abitibi broadcasts out
valid_word is: his
        Nearest to his is: their its her the s recordings alignment dose
valid_word is: two
        Nearest to two is: three four five six one seven eight callithrix
valid_word is: three
        Nearest to three is: four five six two eight seven nine one
valid_word is: but
        Nearest to but is: however and agouti thibetanus dasyprocta michelob meridians victoriae
valid_word is: more
        Nearest to more is: most quarried michelob very microcebus hamburger cl plaintext
valid_word is: the
        Nearest to the is: their its microbats akita dasyprocta a this his
valid_word is: in
        Nearest to in is: at on and from of kapoor thibetanus with
valid_word is: united
        Nearest to united is: pedro roman perimeter primigenius abitibi lemonade functions octavia
valid_word is: during
        Nearest to during is: in at clergyman receivers after for answered chesapeake
Average loss at 61000 step is:4.859219 
Average loss at 62000 step is:4.740433 
Average loss at 63000 step is:4.596152 
Average loss at 64000 step is:4.966679 
Average loss at 65000 step is:5.010026 
valid_word is: state
        Nearest to state is: ramps kapoor meridians michelob predates stadium casement nils
valid_word is: states
        Nearest to states is: stabbed acres scholarships pains expansion imran zubaydah twh
valid_word is: it
        Nearest to it is: he this there she they which akita callithrix
valid_word is: up
        Nearest to up is: trumps granada clo tours extract dasyprocta columbia integrity
valid_word is: between
        Nearest to between is: in with deprived mourned holographic original davids hometown
valid_word is: with
        Nearest to with is: in or fets by dasyprocta between six markov
valid_word is: use
        Nearest to use is: agouti dasyprocta thaler cebus microcebus dek abitibi out
valid_word is: his
        Nearest to his is: their its her the s alignment recordings dose
valid_word is: two
        Nearest to two is: three four six five one seven eight michelob
valid_word is: three
        Nearest to three is: four five six two seven eight nine callithrix
valid_word is: but
        Nearest to but is: however and agouti dasyprocta thibetanus michelob which meridians
valid_word is: more
        Nearest to more is: most very quarried less michelob microcebus plaintext hamburger
valid_word is: the
        Nearest to the is: microbats its their this akita his dasyprocta a
valid_word is: in
        Nearest to in is: at kapoor from during thaler on callithrix initial
valid_word is: united
        Nearest to united is: pedro perimeter roman abitibi primigenius lemonade clement nlm
valid_word is: during
        Nearest to during is: in at clergyman after receivers from chesapeake thibetanus
Average loss at 66000 step is:4.913359 
Average loss at 67000 step is:4.897437 
Average loss at 68000 step is:4.929057 
Average loss at 69000 step is:4.698001 
Average loss at 70000 step is:4.857533 
valid_word is: state
        Nearest to state is: compost ramps stadium meridians states kapoor predates interludes
valid_word is: states
        Nearest to states is: stabbed acres pains michelob scholarships imran expansion cl
valid_word is: it
        Nearest to it is: he this there she they akita callithrix which
valid_word is: up
        Nearest to up is: trumps dasyprocta granada clo tours extract lignite selma
valid_word is: between
        Nearest to between is: in with deprived holographic mourned original davids hometown
valid_word is: with
        Nearest to with is: in or fets by dasyprocta between and purchases
valid_word is: use
        Nearest to use is: agouti dasyprocta thaler cebus microcebus abitibi abet convicted
valid_word is: his
        Nearest to his is: their its her the s recordings megabats alignment
valid_word is: two
        Nearest to two is: three four six one five seven eight callithrix
valid_word is: three
        Nearest to three is: four five six two seven eight callithrix nine
valid_word is: but
        Nearest to but is: however and agouti dasyprocta michelob which thibetanus meridians
valid_word is: more
        Nearest to more is: most very cira less quarried plaintext michelob microcebus
valid_word is: the
        Nearest to the is: microbats their its this akita dasyprocta a cebus
valid_word is: in
        Nearest to in is: at from on kapoor during thaler thibetanus with
valid_word is: united
        Nearest to united is: pedro of perimeter abitibi roman primigenius clement lemonade
valid_word is: during
        Nearest to during is: in at after clergyman receivers chesapeake thibetanus from
Average loss at 71000 step is:4.843493 
Average loss at 72000 step is:4.764801 
Average loss at 73000 step is:4.790531 
Average loss at 74000 step is:4.783862 
Average loss at 75000 step is:4.904189 
valid_word is: state
        Nearest to state is: compost ramps stadium kapoor meridians predates interludes antinomies
valid_word is: states
        Nearest to states is: stabbed acres scholarships pains michelob expansion imran zubaydah
valid_word is: it
        Nearest to it is: he this there she they which akita callithrix
valid_word is: up
        Nearest to up is: trumps tours granada clo lignite extract dasyprocta selma
valid_word is: between
        Nearest to between is: in with holographic deprived mourned original davids hometown
valid_word is: with
        Nearest to with is: in or fets dasyprocta between markov sprague purchases
valid_word is: use
        Nearest to use is: agouti dasyprocta thaler cebus microcebus abitibi dek sprint
valid_word is: his
        Nearest to his is: their its her the s recordings megabats alignment
valid_word is: two
        Nearest to two is: three four six five seven one eight callithrix
valid_word is: three
        Nearest to three is: four six five two seven eight nine callithrix
valid_word is: but
        Nearest to but is: however agouti dasyprocta and michelob meridians thibetanus which
valid_word is: more
        Nearest to more is: most very less quarried cira michelob plaintext microcebus
valid_word is: the
        Nearest to the is: microbats their its akita his ramps dasyprocta agouti
valid_word is: in
        Nearest to in is: at kapoor from on of during thaler and
valid_word is: united
        Nearest to united is: pedro perimeter of roman clement lemonade abitibi antiparticle
valid_word is: during
        Nearest to during is: in at after clergyman from receivers is chesapeake
Average loss at 76000 step is:4.813737 
Average loss at 77000 step is:4.847285 
Average loss at 78000 step is:4.753476 
Average loss at 79000 step is:4.813591 
Average loss at 80000 step is:4.790038 
valid_word is: state
        Nearest to state is: compost ramps kapoor stadium meridians michelob states antinomies
valid_word is: states
        Nearest to states is: stabbed scholarships acres michelob state pains expansion imran
valid_word is: it
        Nearest to it is: he this there she they which akita victoriae
valid_word is: up
        Nearest to up is: trumps tours vec granada kanem dasyprocta clo lignite
valid_word is: between
        Nearest to between is: with in holographic deprived mourned original hometown afl
valid_word is: with
        Nearest to with is: in or between by fets dasyprocta thibetanus callithrix
valid_word is: use
        Nearest to use is: agouti dasyprocta cebus thaler abitibi microcebus dek abet
valid_word is: his
        Nearest to his is: their its her the s recordings megabats prism
valid_word is: two
        Nearest to two is: three four six five seven one eight callithrix
valid_word is: three
        Nearest to three is: four five six two seven eight callithrix microcebus
valid_word is: but
        Nearest to but is: however and agouti dasyprocta michelob meridians thibetanus thaler
valid_word is: more
        Nearest to more is: most very less quarried cira michelob microcebus plaintext
valid_word is: the
        Nearest to the is: its their microbats akita this dasyprocta a his
valid_word is: in
        Nearest to in is: at during kapoor on from thaler nine with
valid_word is: united
        Nearest to united is: pedro of perimeter roman clement abitibi lemonade antiparticle
valid_word is: during
        Nearest to during is: in at after clergyman from receivers was thibetanus
Average loss at 81000 step is:4.822560 
Average loss at 82000 step is:4.770881 
Average loss at 83000 step is:4.769358 
Average loss at 84000 step is:4.790236 
Average loss at 85000 step is:4.782678 
valid_word is: state
        Nearest to state is: ramps compost kapoor stadium states meridians michelob antinomies
valid_word is: states
        Nearest to states is: state stabbed scholarships michelob acres expansion imran viewer
valid_word is: it
        Nearest to it is: he this there she they which akita callithrix
valid_word is: up
        Nearest to up is: trumps tours vec out lignite granada kanem extract
valid_word is: between
        Nearest to between is: with in holographic deprived mourned original hometown afl
valid_word is: with
        Nearest to with is: in tamias between or fets by and dasyprocta
valid_word is: use
        Nearest to use is: agouti dasyprocta cebus thaler abitibi microcebus tamias dek
valid_word is: his
        Nearest to his is: their its her the s recordings megabats altenberg
valid_word is: two
        Nearest to two is: three four one six five seven eight microcebus
valid_word is: three
        Nearest to three is: five four seven two six eight one nine
valid_word is: but
        Nearest to but is: however and agouti dasyprocta michelob meridians which although
valid_word is: more
        Nearest to more is: most very less quarried cotswold michelob cira plaintext
valid_word is: the
        Nearest to the is: its their akita microbats a this dasyprocta cebus
valid_word is: in
        Nearest to in is: at during kapoor on from of and thibetanus
valid_word is: united
        Nearest to united is: of pedro roman perimeter abitibi primigenius antiparticle lemonade
valid_word is: during
        Nearest to during is: in at after clergyman for from receivers chesapeake
Average loss at 86000 step is:4.719234 
Average loss at 87000 step is:4.690412 
Average loss at 88000 step is:4.693576 
Average loss at 89000 step is:4.759110 
Average loss at 90000 step is:4.723030 
valid_word is: state
        Nearest to state is: compost stadium ramps kapoor states meridians antinomies interludes
valid_word is: states
        Nearest to states is: michelob scholarships state stabbed acres imran expansion tamarin
valid_word is: it
        Nearest to it is: he this there she they which akita callithrix
valid_word is: up
        Nearest to up is: trumps tours out lignite granada kanem selma extract
valid_word is: between
        Nearest to between is: with in holographic deprived mourned original hometown afl
valid_word is: with
        Nearest to with is: tamias between in or dasyprocta fets for markov
valid_word is: use
        Nearest to use is: agouti dasyprocta cebus fsm abitibi thaler microcebus methane
valid_word is: his
        Nearest to his is: their her its the s recordings megabats altenberg
valid_word is: two
        Nearest to two is: three four five six seven one eight callithrix
valid_word is: three
        Nearest to three is: five four two six seven eight callithrix microcebus
valid_word is: but
        Nearest to but is: however and agouti which dasyprocta although michelob meridians
valid_word is: more
        Nearest to more is: most less very cotswold cira quarried terribly microcebus
valid_word is: the
        Nearest to the is: akita its microbats dasyprocta ramps their callithrix cebus
valid_word is: in
        Nearest to in is: at during kapoor on and thaler of from
valid_word is: united
        Nearest to united is: of roman pedro perimeter abitibi antiparticle clement nlm
valid_word is: during
        Nearest to during is: in at after clergyman for of from chesapeake
Average loss at 91000 step is:4.704591 
Average loss at 92000 step is:4.708268 
Average loss at 93000 step is:4.578086 
Average loss at 94000 step is:4.664870 
Average loss at 95000 step is:4.694557 
valid_word is: state
        Nearest to state is: ramps compost kapoor stadium meridians states michelob antinomies
valid_word is: states
        Nearest to states is: state scholarships michelob stabbed expansion acres imran viewer
valid_word is: it
        Nearest to it is: he this there she they which but akita
valid_word is: up
        Nearest to up is: trumps out tours him them selma lignite kanem
valid_word is: between
        Nearest to between is: with in holographic deprived mourned original hometown afl
valid_word is: with
        Nearest to with is: in tamias between eight or for fets six
valid_word is: use
        Nearest to use is: agouti dasyprocta fsm cebus abitibi thaler microcebus sprint
valid_word is: his
        Nearest to his is: their her its the s megabats prism recordings
valid_word is: two
        Nearest to two is: three four five six seven one eight callithrix
valid_word is: three
        Nearest to three is: five four two seven six eight callithrix microcebus
valid_word is: but
        Nearest to but is: however and which agouti dasyprocta although though meridians
valid_word is: more
        Nearest to more is: most less very quarried widening cira cotswold terribly
valid_word is: the
        Nearest to the is: its their his a akita microbats this dasyprocta
valid_word is: in
        Nearest to in is: at during kapoor on from thaler within thibetanus
valid_word is: united
        Nearest to united is: roman perimeter pedro antiparticle abitibi clement plum nlm
valid_word is: during
        Nearest to during is: in after at clergyman of from for chesapeake
Average loss at 96000 step is:4.768890 
Average loss at 97000 step is:4.600439 
Average loss at 98000 step is:4.638365 
Average loss at 99000 step is:4.677196 
Average loss at 100000 step is:4.664153 
valid_word is: state
        Nearest to state is: stadium compost ramps kapoor states meridians michelob interludes
valid_word is: states
        Nearest to states is: scholarships state michelob stabbed expansion acres imran calif
valid_word is: it
        Nearest to it is: he this there she they which akita callithrix
valid_word is: up
        Nearest to up is: trumps out tours them him selma lignite kanem
valid_word is: between
        Nearest to between is: with in holographic mourned deprived original hometown davids
valid_word is: with
        Nearest to with is: in tamias between or including and fets markov
valid_word is: use
        Nearest to use is: agouti dasyprocta cebus fsm abitibi thaler microcebus abet
valid_word is: his
        Nearest to his is: their her its the s megabats prism recordings
valid_word is: two
        Nearest to two is: three four six five seven one eight microcebus
valid_word is: three
        Nearest to three is: five four two six seven eight callithrix microcebus
valid_word is: but
        Nearest to but is: however and although agouti though dasyprocta meridians or
valid_word is: more
        Nearest to more is: most less very widening quarried microcebus michelob plaintext
valid_word is: the
        Nearest to the is: its their akita microbats his dasyprocta this agouti
valid_word is: in
        Nearest to in is: during at kapoor from within on thaler thibetanus
valid_word is: united
        Nearest to united is: of perimeter roman pedro abitibi antiparticle clement plum
valid_word is: during
        Nearest to during is: in at after clergyman from following thibetanus within
'''
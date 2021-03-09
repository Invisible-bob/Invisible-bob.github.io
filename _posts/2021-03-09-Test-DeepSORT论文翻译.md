## Simple Online And Realtime Tracking With A Deep Association Metric [DeepSORT]

Code: https://github.com/nwojke/deep_sort

Wojke N, Bewley A, Paulus D. Simple online and realtime tracking with a deep association metric[C]//2017 IEEE international conference on image processing (ICIP). IEEE, 2017: 3645-3649.

SORT/DeepSORT算法解析：https://blog.csdn.net/HaoBBNuanMM/article/details/85555547，https://blog.csdn.net/cdknight_happy/article/details/79731981

Deep SORT多目标跟踪算法代码解析(上) - pprp的文章 - 知乎 https://zhuanlan.zhihu.com/p/133678626

### 摘要

简单在线和实时跟踪（SORT）是一种实用的、关注于简单有效的算法进行多对象跟踪的方法。在本文中，我们融合了目标的表观特征以提高SORT方法的性能。因此我们能够改善有遮挡情况下的目标追踪效果，有效地减少了目标ID跳变的次数。我们将许多计算复杂度置于离线预训练阶段中进行考虑；在这个离线预训练阶段中，我们在大规模行人重识别（ReID）的数据集上学习深度关联度量（deep association metric）。 在实时目标追踪过程中，我们提取目标视觉外观空间中的表观特征进行最近邻匹配。实验评估表明，我们对SORT的改进将目标ID跳变的数量减少了45％，在高帧速率下实现了更好的性能。

##### 关键词——计算机视觉，多目标跟踪，数据关联

### 1 概述

根据目标检测领域的最新进展，通过检测来跟踪（tracking-by-detection）已成为多对象跟踪的一种主要范例。在这种范例中，算法通常在一次处理整个视频批次的全局优化问题中找到对象轨迹。例如网络流公式[1、2、3]和概率图形模型[4、5、6、7]已成为这种方法的流行框架。但是由于batch处理的特点，这些方法不适用于需要在每个时间提供目标ID的在线目标检测场景。更传统的方法是多重假设跟踪（MHT）[8]和联合概率数据关联过滤器（JPDAF）[9]，这些方法逐帧地执行数据关联。在JPDAF中，通过对各个度量的关联可能性进行加权可以生成单状态假设。在MHT中，所有可能的假设都将被跟踪，但是必须进行方案剪枝以保证可计算。JPDAD和MHT两种方法最近都在检测跟踪的场景中被重新研究[10，11]并显示出不错的结果。但是这些方法的性能提升都需要更大的计算量和和实现复杂性。

简单在线和实时跟踪（SORT）[12]是一个更简单的框架，它在图像空间中做**卡尔曼滤波**，并使用**匈牙利方法**逐帧关联数据，使用关联度量（association metric）来度量边界框bbox的重叠。SORT在高帧率下获得了良好的性能。基于使用最先进的检测器[14]，SORT在MOT数据集[13]中的平均排名高于标准检测上的MHT。这个结果强调了目标检测器性能对总体跟踪结果的影响。

虽然在跟踪精度和准确性方面取得了良好的性能，但SORT方法的目标ID切换次数相对较多，这是因为所使用的关联度量只有在状态估计不确定度较低时才是准确的。因此，SORT方法在跟踪被遮挡目标时有一定缺陷，而这些遮挡物通常出现在摄像机的正面视场中。为了克服目标ID切换的问题，我们通过使用结合了运动和外观信息的更有根据的度量，来替换关联度量。进一步地，我们应用了用于大规模行人重识别ReID数据集上的卷积神经网络。 通过ReID CNN的集成，我们提高了算法对漏检和被遮挡问题的鲁棒性，同时使系统能有效地适用于在线场景。

### 2 使用深度关联度量的SORT算法

我们的DeepSORT采用传统的单假设跟踪方法，以及递推卡尔曼滤波和逐帧数据关联的方法。在下一节中我们将更详细地描述本系统的核心组件。

#### 2.1 轨迹处理与状态估计

跟踪轨迹处理和卡尔曼滤波框架与文献[12]中的原始公式基本相同。我们假设了一个非常普遍的跟踪场景，在这种情况下相机是不加标记的，而且未知目标的自我运动信息。这些情况是目前多目标跟踪基准[15]中考虑的最常见的设置。因此，我们的跟踪场景在八维状态空间(u, v, γ, h, x', y', γ', h')上定义，其中包含边界框bbox中心位置(u, v)、纵横比γ、高度h和它们在图像坐标中的速度。我们使用了一个匀速运动的标准卡尔曼滤波器和线性观测模型，其中我们使用边界坐标(u, v. γ. h)直接观察物体状态。

对于每个跟踪过程k，我们计算自上一次成功的测量ak以来的帧数。该帧数计数器在卡尔曼滤波预测期间递增，并在跟踪成功时重置为0。超过预定义的最大范围Amax的跟踪对象将被认为已经离开了画面，并将从重跟踪集合中删除。对于不能与现有轨迹相关联的目标，都会启动新的轨迹假设；这些新的跟踪器在前三帧是试探性的。在这段时间里，我们期望在每一步骤中都有一个成功的测量关联，未成功与其前三帧内的度量相关联的轨迹将被删除。

#### 2.2 分配问题

解决预测的卡尔曼状态与新测量值之间的关联的一种常规方法，是建立可以使用匈牙利算法来解决的分配问题。 在这个问题中，我们通过两个适当的度量组合来整合运动和外观信息。

为了结合运动信息，我们使用距离预测的卡尔曼状态和新测量值之间的马哈拉诺比斯Mahalanobis（平方）距离：

<img src="https://img-blog.csdnimg.cn/20190101201037323.jpg" alt="img" style="zoom:67%;" />

**d1(i, j)**表示第j个检测器和第i条轨迹间的运动匹配度。其中，我们用(yi，Si)表示第i个跟踪器到度量空间的预测，用dj表示第j个检测器bbox的状态。**马氏距离**通过测量检测器与平均轨迹位置的标准差来衡量状态估计的不确定性。【马氏距离计算物体检测Bbox dj和物体跟踪BBox yi之间的距离】此外，利用这一度量d1(i, j)，可以通过在95%置信区间上从逆χ2卡方分布中计算出的马氏距离来排除不相关的情况。定义如下示性函数：

<img src="https://img-blog.csdnimg.cn/20191016145541906.png" alt="img" style="zoom:67%;" />

如果第i个跟踪轨迹和第j个检测之间的距离是关联的，则结果为1。【两者距离≤特定阈值t(1)，则表示两者关联】对于我们的四维测量空间(w,v,r,h)，对应的Mahalanobis阈值为t(1)= 9.4877。

当运动不确定性较低时，马氏距离是一种合适的关联度量。但在我们的图像空间中，从卡尔曼滤波框架预测的状态分布仅提供了对象位置的粗略估计。此外，无法解释的摄像机运动可能会在图像平面中引入快速位移，从而使马氏距离不适合作为跟踪被遮挡目标的合适度量，不能很好的解决物体被长时间遮挡后关联不正确导致**ID Switch**的问题。

因此我们将第二个指标集成到分配问题中。对于每个边界框bbox检测dj，我们计算|| rj || = 1的外观描述符rj。此外，对于每个轨迹k，我们保留最后Lk = 100个成功跟踪的相关外观描述符的集合{Rk}。定义第二个度量**d2(i, j)**将测量外观空间中第i个轨迹和第j个检测器之间的**最小余弦距离**：

<img src="https://img-blog.csdnimg.cn/20191016145931357.png" alt="img" style="zoom:67%;" />

**d2(i, j)**表示第i个物体跟踪的所有特征向量和第j个物体检测之间的最小余弦距离。同样的，我们引入一个二进制变量来指示根据该指标是否允许关联【两者距离≤特定阈值t(2)，则表示两者关联】：

<img src="https://img-blog.csdnimg.cn/20191016145931379.png" alt="img" style="zoom:67%;" />

我们在单独的训练数据集上找到了合适的阈值t(2)。我们应用预训练的CNN来计算边界框外观描述符(box appearance descriptors)。该网络的体系结构在本文2.4节中描述。

In combination, both metrics complement each other by serving different aspects of the assignment problem. On the one hand, the Mahalanobis distance provides information about possible object locations based on motion that are particularly useful for short-term predictions. On the other hand, the cosine distance considers appearance information that are particularly useful to recover identities after longterm occlusions, when motion is less discriminative. To build the association problem we combine both metrics using a weighted sum.

以上两个度量指标作为分配问题的不同方面，可以结合起来、相互补充。一方面，**马氏距离基于目标的运动提供可能的目标位置信息**，这对于短时预测特别有用。另一方面，**当目标运动难以区别是时，余弦距离会考虑外观信息**，这些信息对于长时间遮挡后重新获得目标ID特别有用。为了建立度量的关联问题，我们使用加权和将两个指标结合起来【作为计算第i个物体跟踪和第j个物体检测之间关联度量的总公式】：

<img src="https://img-blog.csdnimg.cn/20191016145931410.png" alt="img" style="zoom: 50%;" />（5）

组合距离阈值判断不等式，作为总的判断第i个物体跟踪和第j个物体检测之间的距离（关联度量）是否关联的总公式：

<img src="https://img-blog.csdnimg.cn/20191016145931408.png" alt="img" style="zoom:50%;" />（6）

The influence of each metric on the combined association cost can be controlled through hyperparameter λ . During our experiments we found that setting λ= 0 is a reasonable choice when there is substantial camera motion. In this setting, only appearance information are used in the association cost term.However, the Mahalanobis gate is still used to disregarded infeasible assignments based on possible object locations inferred by the Kalman filter.

引入超参数λ来控制每个度量指标对关联成本的影响。在我们的实验过程中，我们发现当摄像机运动幅度较大时，设置λ=0是合理的选择。在此设置中，关联费用项中仅使用外观信息；不过基于由卡尔曼滤波器推断出的可能的物体位置，马氏距离仍然被用来忽略不可行的分配。

#### 2.3 关联匹配

为了解决全局分配问题中的测量与跟踪过程的关联，我们设计了一个级联算法来解决一系列子问题。考虑以下情况：当物体被长时间遮挡时，后续的卡尔曼滤波预测的关于目标位置的不确定性就会大大增加，状态空间内的可观察性就会大大降低。直观地来看，关联度量应通过增加测量与跟踪过程的距离，往往遮挡时间较长的那条轨迹因为长时间未更新位置信息，追踪预测位置的不确定性更大，即协方差会更大。但违反直觉的是，当两个跟踪器竞争同一检测框时，马氏距离会带来较大的不确定性，因为马氏距离有效地减小了任何检测的标准偏差与映射轨迹均值之间的距离（马氏距离计算时使用了协方差的倒数，因此马氏距离会更小，使得检测结果更可能和遮挡时间较长的那条轨迹相关联）。这种现象应该避免，因为它可能破坏追踪的持续性。因此我们引入了一个级联匹配算法，级联匹配给予常见对象更高的匹配优先级，这样每次匹配的时候考虑的都是遮挡时间相同的轨迹。

【核心思想：由小到大对消失时间相同的轨迹进行匹配，这样首先保证了对最近出现的目标赋予最大的优先权】

<img src="https://img-blog.csdn.net/20180331160629759?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Nka25pZ2h0X2hhcHB5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="这里写图片描述" style="zoom:50%;" />

列表1概述了我们的级联匹配算法。我们给出物体跟踪集合T和物体检测集合D，以及最大age Amax作为输入。

第1-2行中，我们计算了关联成本矩阵(公式5)和是否关联的矩阵(公式6)，C矩阵存放所有物体跟踪i与物体检测j之间距离的计算结果，B矩阵存放所有物体跟踪i与物体检测j之间是否关联的判断(0/1)。

第5行对目标轨迹n进行迭代（从刚刚匹配成功的跟踪器循环遍历到最多已经有Amax次没有匹配的跟踪器），以解决随着age增长的轨迹的线性分配问题。在第6行中，选择在最近n帧中未与检测框相关联的跟踪框Tn的子集。在第7行中，解决了Tn中的跟踪框与不匹配的检测框U之间的线性分配问题。

在第8行和第9行中，我们更新了匹配项M和未匹配的检测框集合U，并在第11行中完成后返回。【此匹配级联将优先考虑年龄较小的跟踪器，即最近匹配成功的跟踪器】

在最后的匹配阶段，我们对unconfirmed和age=1的未匹配轨迹和检测目标进行基于IoU的匹配。这可以缓解因为表观突变或者部分遮挡导致的较大变化，并提高了针对错误初始化的鲁棒性。

#### 2.4 深度特征描述器

<img src="https://img-blog.csdn.net/20180331161530956?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Nka25pZ2h0X2hhcHB5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="这里写图片描述" style="zoom: 33%;" />

在实际的在线跟踪应用之前，我们方法要求将具有良好区分性的功能嵌入脱机训练，具体方法是通过使用简单的最近邻查询而不进行额外的度量学习。为此，我们采用了经过大规模ReID数据集[21]训练的CNN，该数据集包含1,100,000张、1,261名行人的图像。
表1中显示了该CNN网络的结构。我们使用了一个较宽的残差网络residual network[22]，该网络具有两个卷积层和六个残差块。网络的第10层Dense 10计算了128维的全局特征图。（最后一层的）final batch和l2归一化将目标特征投影到单位超球面上，与我们的余弦外观度量兼容。该网络总共具有2,800,864个参数，在GTX1050上单次32个边界框bbox的正向传播大约需要30 ms。因此该网络非常适合在线跟踪。



### 3 实验

我们在MOT16基准[15]上评估DeepSORT跟踪器的性能。该基准评估了在七个测试序列上的跟踪性能，包括具有移动摄像机的前视场景，以及自上而下的监视设置。我们使用Yu等人[16]提供的检测结果作为对跟踪器的输入，他们已经在一组公共和私有数据集上训练了Faster RCNN，提供了出色的性能。为了进行公平的比较，我们将SORT进行了相同测试。

评估测试序列的超参数设为λ= 0和Amax = 30帧。如文献[16]中所述，检测阈值的置信度为0.3。我们的方法的其余参数都由MOT16基准提供的单独训练序列提供。具体根据以下度量指标开展：

- 多目标跟踪准确度（MOTA, Multi-object tracking accuracy）：所有关于误判正、误判负和ID switch的总体跟踪accuracy之和[23]。
- 多目标跟踪精度（MOTP, Multi-object tracking precision）：在边界框重叠方面的所有跟踪精度[23]。
- 最常跟踪（MT, Mostly tracked）：实地跟踪中，至少80％的跟踪周期中使用相同的标签的目标百分比。
- 最常丢失（ML, Mostly lost）：实地跟踪中，最多只能追踪到跟踪周期的20％的目标百分比。
- 标签切换（ID, Identity switches）：报告的ID的次数-真实轨道的真实性发生了变化。
- 碎片（FM, Fragmentation）：由于漏检missing detection导致的中断跟踪的次数

<img src="https://img-blog.csdnimg.cn/20190118154820851.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Nka25pZ2h0X2hhcHB5,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 50%;" />

表2为评估结果，我们的调整成功地减少了 ID切换的数量。与SORT相比， ID switches减少了约45％。与此同时由于需要再遮挡和漏检的场景下来保持目标的ID，轨迹碎片会稍微增加。我们也注意到最常跟踪对象MT的数量显着增加，而最常丢失对象ML的数量则减少了。总体而言，由于外观信息的整合，我们能在更长被遮挡的情况成功跟踪目标。

DeepSORT方法也可以和其他在线跟踪框架强力竞争。DeepSORT在所有在线方法中得到了最少的ID切换次数，同时保持了竞争性的多目标跟踪准确度MOTA分数、跟踪碎片和误判负情况。多目标跟踪准确性主要受到大量误检（flase positives）的影响，鉴于它们对MOTA分数的总体影响，对检测应用较大的置信度阈值可能会大大提高我们算法的性能。但是对跟踪输出图像进行的人工检查表明，这些误检主要是由静态场景中的零星检测器（sporadic detector）响应产生的。由于我们的最大允许跟踪周期Amax比较大，这些误检通常会与对象轨迹结合在一起；同时我们没有观察到在误检ID之间频繁跳跃的轨迹。跟踪器反而通常会在报告的对象位置生成相对稳定的固定轨道。

DeepSORT的应用以大约20 Hz的频率运行，其中约一半时间用于特征生成。因此若使用GPU运行，该系统可以保持实时运行的效率。



### 4 总结

我们在SORT算法的基础上做了进一步扩展，通过预先训练的关联指标整合了外观信息。这我们能够有效跟踪被遮挡时间较长的目标，从而使SORT能与最新在线跟踪算法一较高下。同时该算法仍然易于实现，并且可以实时运行。



### 5 参考文献

[1]  L. Zhang, Y. Li, and R. Nevatia, “Global data associa- tion for multi-object tracking using network flows,” in *CVPR*, 2008, pp. 1–8.

[2]  H. Pirsiavash, D. Ramanan, and C. C. Fowlkes, “Globally-optimal greedy algorithms for tracking a vari- able number of objects,” in *CVPR*, 2011, pp. 1201– 1208.

[3]  J. Berclaz, F. Fleuret, E. Tu ̈retken, and P. Fua, “Multi- ple object tracking using k-shortest paths optimization,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 33, no. 9, pp. 1806–1819, 2011.

[4]  B. Yang and R. Nevatia, “An online learned CRF model for multi-target tracking,” in *CVPR*, 2012, pp. 2034– 2041.

[5]  B. Yang and R. Nevatia, “Multi-target tracking by on- line learning of non-linear motion patterns and robust appearance models,” in *CVPR*, 2012, pp. 1918–1925.

[6]  A. Andriyenko, K. Schindler, and S. Roth, “Discrete- continuous optimization for multi-target tracking,” in *CVPR*, 2012, pp. 1926–1933.

[7]  A. Milan, K. Schindler, and S. Roth, “Detection- and trajectory-level exclusion in multiple object tracking,” in *CVPR*, 2013, pp. 3682–3689.

[8]  D.B.Reid,“Analgorithmfortrackingmultipletargets,” *IEEE Trans. Autom. Control*, vol. 24, no. 6, pp. 843– 854, 1979.

[9]  T.E. Fortmann, Y. Bar-Shalom, and M. Scheffe, “Sonar tracking of multiple targets using joint probabilistic data association,” *IEEE J. Ocean. Eng.*, vol. 8, no. 3, pp. 173–184, 1983.

[10]  C. Kim, F. Li, A. Ciptadi, and J. M. Rehg, “Multiple hypothesis tracking revisited,” in *ICCV*, 2015, pp. 4696– 4704.

[11]  S.H.Rezatofighi,A.Milan,Z.Zhang,Qi.Shi,An.Dick, and I. Reid, “Joint probabilistic data association revis- ited,” in *ICCV*, 2015, pp. 3047–3055.

[12]  A. Bewley, G. Zongyuan, F. Ramos, and B. Upcroft, “Simple online and realtime tracking,” in *ICIP*, 2016, pp. 3464–3468.

[13]  L. Leal-Taixe ́, A. Milan, I. Reid, S. Roth, and K. Schindler, “MOTChallenge 2015: Towards a bench- mark for multi-target tracking,” *arXiv:1504.01942 [cs]*, 2015.

[14] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-time object detection with region proposal networks,” in *NIPS*, 2015.

[15] A. Milan, L. Leal-Taixe ́, I. Reid, S. Roth, and K. Schindler, “Mot16: A benchmark for multi-object tracking,” *arXiv preprint arXiv:1603.00831*, 2016.

[16] F. Yu, W. Li, Q. Li, Y. Liu, X. Shi, and J. Yan, “Poi: Multiple object tracking with high performance detec- tion and appearance feature,” in *ECCV*. Springer, 2016, pp. 36–42.

[17] M. Keuper, S. Tang, Y. Zhongjie, B. Andres, T. Brox, and B. Schiele, “A multi-cut formulation for joint segmentation and tracking of multiple objects,” *arXiv preprint arXiv:1607.06317*, 2016.

[18] B. Lee, E. Erdenee, S. Jin, M. Y. Nam, Y. G. Jung, and P. K. Rhee, “Multi-class multi-object tracking using changing point detection,” in *ECCV*. Springer, 2016, pp. 68–83.

[19] W. Choi, “Near-online multi-target tracking with aggre- gated local flow descriptor,” in *ICCV*, 2015, pp. 3029– 3037.

[20] R.Sanchez-Matilla,F.Poiesi,andA.Cavallaro,“Online multi-target tracking with strong and weak detections,” in *European Conference on Computer Vision*. Springer, 2016, pp. 84–99.

[21] L. Zheng, Z. Bie, Y. Sun, J. Wang, C. Su, S. Wang, and Q. Tian, “MARS: A video benchmark for large-scale person re-identification,” in *ECCV*, 2016.

[22] S. Zagoruyko and N. Komodakis, “Wide residual net- works,” in *BMVC*, 2016, pp. 1–12.

[23] K. Bernardin and R. Stiefelhagen, “Evaluating mul- tiple object tracking performance: The CLEAR MOT metrics,” *EURASIP J. Image Video Process*, vol. 2008, 2008.

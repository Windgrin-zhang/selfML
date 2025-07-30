<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Machine Learning Notes</title>
  <!-- MathJax -->
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
    h1, h2, h3 { color: #333; }
    pre { background: #f4f4f4; padding: 10px; overflow: auto; }
    blockquote { border-left: 4px solid #ccc; margin: 10px 0; padding-left: 10px; color: #666; }
    ul { margin: 10px 0; padding-left: 20px; }
  </style>
</head>
<body>
  <h1>Machine Learning</h1>
  <h2>knn代码练习</h2>
  <p>因为不喜欢命令行运行，将所有路径改为相对路径</p>
  <h3>simple_knn</h3>
  <ul>
    <li>简单二维分类</li>
    <li>使用欧氏距离计算test与数据集差值，取k个最小距离依次按类别投票</li>
  </ul>
  <h3>海伦约会</h3>
  <ul>
    <li>简单三维分类</li>
    <li>同样使用欧氏距离，使用相同权值，归一化到0-1区间</li>
  </ul>
  <h3>手写数字识别</h3>
  <ul>
    <li>原始输入32*32 更改输入维度（1，1024）</li>
    <li>label为每个txt文件的名字，203个训练集，88个验证集（个人认为比例不妥）</li>
    <li>错误率大概在0.0125%左右</li>
    <li>不如我在Mnist上的成果：1、训练量少 2、训练模型方式较简单 3、数据集分配比例</li>
  </ul>
  <h2>Decision Tree 决策树代码练习</h2>
  <p>感觉代码构建复杂得要死，适合简单的区分。</p>
  <h3>构建结构</h3>
  <blockquote>
    个人对构建理解为：信息熵增益效果即为权重的选择，越重要的放得越靠近根节点，增益值越小影响越大，影响最大的必然放在根节点
  </blockquote>
  <ul>
    <li>决策树训练使用流程：</li>
    <li>基本不需要数据预处理，只需要对种类数量的判断是否足够训练、种类是否重合等简单情况要处理</li>
    <li>先寻找每类分割阈值，子集尽可能纯（ID3、Gini、C4.5等）</li>
  </ul>
  <blockquote>
    <strong>ID3</strong><br>
    $$
    Gain(D, A) = Entropy(D) - \sum_{v \in values(A)} \frac{|D_v|}{|D|} Entropy(D_v)
    $$<br>
    选择信息增益最大的划分属性
  </blockquote>
  <hr>
  <blockquote>
    <strong>Gini (CART)</strong><br>
    $$
    Gini(D) = 1 - \sum_{i=1}^{k} p_i^2
    $$<br>
    为 0 时最纯
  </blockquote>
  <hr>
  <blockquote>
    <strong>C4.5</strong><br>
    $$
    GainRatio(D, A) = \frac{Gain(D, A)}{IV(A)}
    $$<br>
    用于修正信息增益对多值属性的偏好
  </blockquote>
  <ul>
    <li>之后递归构建决策树，满足在左，不满足在右之类</li>
    <li>使用时根据阈值来走即可</li>
  </ul>
</body>
</html>

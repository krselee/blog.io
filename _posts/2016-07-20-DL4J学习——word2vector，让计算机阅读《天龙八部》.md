---
title: DL4J学习——word2vector，让计算机阅读《天龙八部》
tag: tech
---

很早在实验室就看见钱宝宝用Google的Word2Vector来跑《天龙八部》，找出与指定词最相关的几个词，最近正好学习新出的深度学习开源项目DeepLearning4J，于是就拿这个例子来练手吧。DL4J的详细用法请看 [DL4J快速入门](http://deeplearning4j.org/quickstart.html) 。       
DeepLearning4J的Example中自带了很多应用实例，Word2Vector也在其中，因此我的工作主要是以下几步：        

1. 准备开发环境和原始数据
2. 分词，格式转换
3. 构建Word2Vector模型并训练
4. 测试并输出
 
## 一．准备开发环境和原始数据
开发环境我使用的是IDEA（用eclipse也OK），JDK1.7，Maven3.3.1。
上武侠小说网下载一篇《天龙八部》，去掉文件首尾的不相关信息，重命名放到指定位置，OK。

## 二．分词、格式转换
我比较喜欢使用复旦NLP，一是用惯了熟练，二是使用起来也方便，Maven引用FNLP有一点小问题，用官方给的maven坐标不能，解决方法可以参考我[以前的文章](http://blog.csdn.net/a398942089/article/details/51152776)，这里不再赘述。
新建Java工程（或者直接使用DL4J-example工程），新建JavaClass，命名为FudanTokenizer：


```
package edu.zju.cst.krselee.example.word2vector;  

/** 
 * Created by KrseLee on 16/7/20. 
 */  
  
import org.fnlp.nlp.cn.tag.CWSTagger;  
import org.fnlp.util.exception.LoadModelException;  
  
import java.io.IOException;  
import java.util.List;  
  
import org.fnlp.ml.types.Dictionary;  
import org.fnlp.nlp.corpus.StopWords;  

public class FudanTokenizer {  
    private CWSTagger tag;  
  
    private StopWords stopWords;  
  
    public FudanTokenizer() {  
        String path = this.getClass().getClassLoader().getResource("").getPath();  
        System.out.println(path);  
        try {  
            tag = new CWSTagger(path + "models/seg.m");  
        } catch (LoadModelException e) {  
            e.printStackTrace();  
        }  
  
    }  
  
    public String processSentence(String context) {  
        String s = tag.tag(context);  
        return s;  
    }  
  
    public String processSentence(String sentence, boolean english) {  
        if (english) {  
            tag.setEnFilter(true);  
        }  
        return tag.tag(sentence);  
    }  
  
    public String processFile(String filename) {  
        String result = tag.tagFile(filename);  
  
        return result;  
    }  
  
    /** 
     * 设置分词词典 
     */  
    public boolean setDictionary() {  
        String dictPath = this.getClass().getClassLoader().getResource("models/dict.txt").getPath();  
  
        Dictionary dict = null;  
        try {  
            dict = new Dictionary(dictPath);  
        } catch (IOException e) {  
            return false;  
        }  
        tag.setDictionary(dict);  
        return true;  
    }  
  
    /** 
     * 去除停用词 
     */  
    public List<String> flitStopWords(String[] words) {  
        try {  
            List<String> baseWords = stopWords.phraseDel(words);  
            return baseWords;  
        } catch (Exception e) {  
            e.printStackTrace();  
            return null;  
        }  
    }  
}  
```

并将模型文件（可以从[FNLP的release页面](https://github.com/FudanNLP/fnlp/releases)下载）拷入到resources目录下。          
在maven的pom.xml里面添加FNLP的依赖：

```
<dependency>  
    <groupId>org.fnlp</groupId>  
    <artifactId>fnlp-core</artifactId>  
    <version>2.1-SNAPSHOT</version>  
</dependency>  
  
<dependency>  
    <groupId>junit</groupId>  
    <artifactId>junit</artifactId>  
    <version>4.11</version>  
</dependency>  
```

等Maven把工程编译好，将之前下载的数据文件放到resources目录下，新建一个主方法或者单元测试来执行分词：
 
```
public void processFile() throws Exception{  
       String filePath = this.getClass().getClassLoader().getResource("text/tlbb.txt").getPath();  
       BufferedReader in = new BufferedReader(new FileReader(filePath));  
  
       File outfile = new File("/Users/KrseLee/dataset/tlbb_t.txt");  
       if (outfile.exists()) {  
           outfile.delete();  
       }  
       FileOutputStream fop = new FileOutputStream(outfile);  
  
       // 构建FileOutputStream对象,文件不存在会自动新建  
       String line = in.readLine();  
       OutputStreamWriter writer = new OutputStreamWriter(fop, "UTF-8");  
       while(line!=null) {  
           line = tokenizer.processSentence(line);  
           writer.append(line);  
           line = in.readLine();  
       }  
       in.close();  
       writer.close(); // 关闭写入流,同时会把缓冲区内容写入文件  
       fop.close(); // 关闭输出流,释放系统资源  
   }  
```

## 三．构建Word2Vector模型并训练
引入DeepLearning4J的依赖包，新建Java Class ZhWord2Vector，代码如下：

```
package edu.zju.cst.krselee.example.word2vector;  
  
import org.canova.api.util.ClassPathResource;  
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;  
import org.deeplearning4j.models.word2vec.Word2Vec;  
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;  
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;  
import org.slf4j.Logger;  
import org.slf4j.LoggerFactory;  
  
import java.util.Collection;  
  
/** 
 * Created by KrseLee on 16/7/20. 
 */  
public class ZhWord2Vector {  
    private static Logger log = LoggerFactory.getLogger(ZhWord2Vector.class);  
  
    public static void main(String[] args) throws Exception {  
  
        String filePath = new ClassPathResource("text/tlbb_t.txt").getFile().getAbsolutePath();  
  
        log.info("Load & Vectorize Sentences....");  
        // Strip white space before and after for each line  
        SentenceIterator iter = new BasicLineIterator(filePath);  
        // Split on white spaces in the line to get words  
  
        log.info("Building model....");  
        Word2Vec vec = new Word2Vec.Builder()  
            .minWordFrequency(5)  
            .iterations(1)  
            .layerSize(100)  
            .seed(42)  
            .windowSize(5)  
            .iterate(iter)  
            .build();  
  
        log.info("Fitting Word2Vec model....");  
        vec.fit();  
  
        log.info("Writing word vectors to text file....");  
  
        // Write word vectors  
        WordVectorSerializer.writeWordVectors(vec, "tlbb_vectors.txt");  
        WordVectorSerializer.writeFullModel(vec,"tlbb_model.txt");  
        String[] names = {"萧峰","乔峰","段誉","虚竹","王语嫣","阿紫","阿朱","木婉清"};  
        log.info("Closest Words:");  
  
        for(String name:names) {  
            System.out.println(name+">>>>>>");  
            Collection<String> lst = vec.wordsNearest(name, 10);  
            System.out.println(lst);  
        }  
    }  
}  
```

将上一步得到的输出的分词后的小说文件拷贝到resources目录下，准备工作就完成了。

##四．测试并输出
更改你想要查看的单词，运行程序，等待约4分钟，得到输出。不同的电脑因性能原因需要的时间不一致，深度网络的训练本身也是一件费时费力的事情。

```
萧峰>>>>>>  
[段誉, 叫骂, 一队队, 军官, 将, 狗子, 长矛, 指挥, 说, 传令]  
乔峰>>>>>>  
[南, 大侠, 北, 大英雄, 四海, 厮, 听说, 奸谋, 威震, 全舵]  
段誉>>>>>>  
[萧峰, 虚竹, 向, 玄渡, 等, 叫骂, 去, 辽兵, 一边, 城门]  
虚竹>>>>>>  
[段誉, 向西, 萧峰, 向, 城门, 叫骂, 等, 辽兵, 玄鸣, 去]  
王语嫣>>>>>>  
[巴天石, 钟灵, 木婉清, 草海, 朱丹臣, 老婆婆, 瘴气, 贾老者, 嗒嗒嗒, 途中]  
阿紫>>>>>>  
[道, 穆贵妃, 抿嘴笑, 姊夫, 来, 叫, 又, 小嘴, 大人, 什么]  
阿朱>>>>>>  
[深情, 想起, 换上, 父母, 想念, 恩情, 胡作非为, 迫, 情意, 永远]  
木婉清>>>>>>  
[钟灵, 朱丹臣, 巴天石, 秦红棉, 范骅, 一行人, 王语嫣, 墙外, 阮星竹, 巴天]  
```

好了，大功告成。
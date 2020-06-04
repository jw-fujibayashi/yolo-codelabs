id: yolo
satus: making
summary: yolov3

# tensorflow2.0 YOLOv3

## 初めに
Duration: 10

本スライドは、物体検出のアルゴリズムの1つ、YOLOv3をtensorflow2.0で試す手順をまとめたものです。

もちろん既存のデータセットではなく、みなさんが用意したデータを学習させて物体検出を行います。

以下の画像は、「グー」「チョキ」「パー」の手の形を学習させて検出させている様子です。画像などを用意するのは単純作業で辛いですが、上手く検出できるとその反面嬉しさも倍増するので頑張りましょう。

![img1](img_original/img1.jpeg)

### 今回使用するサービスやツール

#### Google Colaboratory

![img2](img_original/img2.jpeg)

Googleが提供する「機械学習/Deep Learningの教育や研究の促進を目的としたプロジェクト」です。ブラウザから操作できるので、環境構築不要です。また、GPUというハイスペックな環境を使用できます。Googleアカウントがない方は取得しておいてください。

Google Colaboratoryで使用するファイルを「ノートブック」と呼び、今回はこちらが用意したノートブックを使用します。

![img3](img_original/img3.jpeg)

このように、ノートブックにはプログラミング言語だけではなくドキュメントも記すことができます。

![img4](img_original/img4.jpeg)

また、セル単位で実行するコードを分けることができます。環境や変数などは同一ノートブック内では共通されます。

コードを実行するには、実行したいセルにカーソルを合わせた状態で、`ctrl`+`Enter`(Macの場合は、`cmd`+`Enter`)です。もしくは、再生ボタンをクリックします。

Google Colaboratoryについて気になる方は公式の[説明](https://colab.research.google.com/notebooks/intro.ipynb?hl=ja)もご参照ください。

#### ラズベリーパイ

![img6](img_original/img6.jpeg)

「ラズベリーパイ」は、必要最低限な機能を備えた小型PCです。その分スペックは低いですが数千円で購入できる手軽さから、子ども用のプログラミング入門キットとしてだけでなく、大学の工学部や情報学部などでの授業に取り入れられてます。また大人も楽しめる趣味として世界中で幅広い層に利用されています。

今回はこのラズパイにカメラをつけて、みなさんの手元でリアルタイム物体検出を行います。

小型なため電力さえあれば設置場所にも困りません。また、ぬいぐるみなどに装着することもできます。

#### VoTT

![img7](img_original/img7.jpeg)

「VoTT」は、Microsoft社が提供する、動画・画像に対してアノテーションをおこなう無償のアノテーションツールです。

アノテーションとは、対象となるデータに対して正解ラベル（タグ）や対象物の座標等関連する情報を注釈として付与することを指します。たとえば、犬の画像に対して「犬」とラベル付けたり、バウンディングボックスで犬を囲んだりします。

---

### 大まかな物体検出の流れ

1. 学習データの作成
    - 検出したいモノの画像用意
    - アノテーション作業
2. 学習
3. 推論（検出）

オリジナルデータで物体検出をするために、まずは学習データを皆さんに作成してもらいます。

その後、上述したGoogle Colaboratoryで学習を行い、最後にラズパイでリアルタイム画像検出を行います。

### 今回のポイント

ラズパイで物体検出を簡単に試せるよう、以下の3つの工夫をしました。

- 転移学習
- YOLOv3-tinyの採用
- 画像の水増し

#### 転移学習

物体検出は、AIに学習させる時間や必要なデータ数が大量になります。そのため、今回は「転移学習」と呼ばれるものを行い「時間」「データ数」の必要量を減らします。

簡単に言うと、ある領域で学習させたモデルを別の領域に適応させる技術です。事前に学習されているため、学習に時間がかかりません。たとえば、「犬の種類の画像認識を行うモデル」を「猫の種類の画像認識」に応用できます。

#### YOLOv3-tinyの採用

今回は、YOLOv3-tinyと呼ばれる「正確さよりもリアルタイム性や軽量さを重視したモデル」を使用します。

精度は落ちますが、ラズパイでも動作できる程度に軽いモデルです。

#### 学習データの水増し

画像認識でもよく採用されている「学習データの水増し」を使って、必要な画像枚数を少なくします。また、この後に作成するアノテーションデータ自体の水増しも行右ことでアノテーション作業量を減らします。

「水増し」という言葉の印象はあまりよくありませんが、単純に学習データとなる画像をプログラムを使って複製するだけです。たとえば、元の画像枚数が20枚しかなくても100枚に増やして学習させることができます。

自分が試したところ、1つのクラスに対して20種類ぐらいの画像がを用意すれば上手く検出してくれました。（最終的には、1つのクラスに対して300 ~ 400枚使用していることにあります）

### 事前準備

いくつか時間のかかる作業があるのでそちらを先に済ませます。

#### ノートブックのコピー

まずは、今回用意したノートブックをみなさんのGoogle Driveにコピーします。

[こちら](https://colab.research.google.com/drive/1FzwCZIIsqvSDktRDWVclO_jokT55qoA0?usp=sharing)をクリックして開いてください。

![img8](img_original/img8.jpeg)

開いたら「ファイル」 > 「ドライブにコピーを保存」をクリックします。これでみなさんのドライブにコピーが作成されます。

![img9](img_original/img9.jpeg)

ただし、このコピーししたノートブックは実行できる状態ではないので「ファイル」>「Playgroundモードで開く」をクリックして実行できるようします。

コピーできたら、1つ目のセル「A.Google Colaboratoryテスト用コード」を試しに実行してみましょう！実行するには、`ctrl`+`Enter`(Macの場合は、`cmd`+`Enter`)です。もしくは、再生ボタンをクリックします。

![img10](img_original/img10.jpeg)

このようになれば問題ありません。

#### Google Driveとの連携

無事にノートブックをコピーできて、実行できるようになったら作業を進めていきます。まずは、Google DriveとGoogle Colaboratoryを連携させます。

「B.Google Driveとの連携」を実行しましょう。

![img11](img_original/img11.jpeg)

実行すると、このようにリンクが現れるのでクリックして開きます。

![img12](img_original/img12.jpeg)

クリックすると、連携先のGoogleアカウントを選びます。複数のアカウントを持っている方は、ノートブックをコピーしたアカウントを選んでください。

![img13](img_original/img13.jpeg)

「許可」を押します。

![img14](img_original/img14.jpeg)

赤色で隠されている部分をコピーします。赤枠箇所をクリックすると自動でコピーされます。

![img15](img_original/img15.jpeg)

コピーできたら、ノートブックに戻ってこの箇所に貼り付けて、`Enter`を押します。

![img16](img_original/img16.jpeg)

少し時間が経って、このように表示されると連携の終了です。

![img17](img_original/img17.jpeg)

左側のフォルダマークをクリックすると、`drive`フォルダーが表示され、ここでdrive内のファイルを確認できます。

#### Gitプロジェクトのダウンロード

今回使用するソースコードをダウンロードします。「C.gitプロジェクトをダウンロード」を実行しましょう。

実行すると、Google Driveの`Colabo NoteBooks`というフォルダーの中に`yolo-tf2-test`というフォルダーが自動でダウンロードされます。

![img18](img_original/img18.jpeg)

このようにDrive内の`Colab Notebooks`フォルダー内にダウンロードされます。

#### 重みのダウンロードと変換

転移学習で使用する「重み」をダウンロードし変換します。ダウンロードには時間がかかります。
「D.重みのダウンロードと変換」を実行しましょう。

実行するとtinyモデルの重みをダウンロードします。もし通常のYOLOの重みも試してみたい方は、

```python=
!wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolo_weight/yolov3-tiny.weights
!python convert.py --weights ./data/yolo_weight/yolov3-tiny.weights --output ./data/weight/yolov3-tiny.tf --tiny

# tinyを使わない場合は以下
# !wget https://pjreddie.com/media/files/yolov3.weights -O data/yolo_weight/yolov3.weights
# !python convert.py --weights ./data/yolo_weight/yolov3.weights --output ./data/weight/yolov3.tf
```
1,2行目をコメントにして、5,6行目をコメントアウトしてください。

```python=
# !wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolo_weight/yolov3-tiny.weights
# !python convert.py --weights ./data/yolo_weight/yolov3-tiny.weights --output ./data/weight/yolov3-tiny.tf --tiny

# tinyを使わない場合は以下
!wget https://pjreddie.com/media/files/yolov3.weights -O data/yolo_weight/yolov3.weights
!python convert.py --weights ./data/yolo_weight/yolov3.weights --output ./data/weight/yolov3.tf
```

これで事前準備はお終いです。ダウンロードには時間がかかるので、終わるのを待つ必要はありません。次のスライドに進みましょう！

## YOLOv3とは
Duration: 10

それでは、今回試す「YOLO」について簡単に解説したいと思います。

難しい内容ですが、理解できなくてもプログラムを実行して物体検出を体験できますので安心してください。そういうものなのか、という程度で読み流してもらえればと思います。

YOLOとは、物体検出の手法の1つです。「v3」というのは、そのままバージョン3のことを表します。（※2020年5月にv4が考案されたようです）物体領域検出と画像認識を同時に行うことでより速く物体検出を行えるのが特徴です。その反面、精度は他のアルゴリズムと比べると少し落ちています。

まずは、YOLOがどのように物体検出を行っているのか簡単なイメージで解説します。v3での改善点は最後に解説します。

![img19](img_original/img19.jpeg)

（https://pjreddie.com/ から引用）

YOLOでは、まず対象の画像をいくつかのセルに分けます。この各セルに対して、「物体領域検出（どこに物体があるのか）」と「画像認識（そのセルが何なのか）」を行います。

![img20](img_original/img20.jpeg)

（https://pjreddie.com/ から引用）

上の画像が「物体領域検出」の様子です。このように、サイズの異なる複数のバウンディングボックスを用意し、それぞれに対して「物体がどれぐらい存在しているのか」を計算します。これを「信頼度スコア」と言います。
物体が存在していなければ、信頼度スコアは0になります。

![img21](img_original/img21.jpeg)  

（https://pjreddie.com/ から引用）

上の画像が「画像認識」の様子です。各セルに対して画像認識を行い、そのセルがどういった画像なのかを計算します。

<!-- ![](https://i.imgur.com/69exYVz.jpeg) -->
![img22](img_original/img22.jpeg)

（https://pjreddie.com/ から引用）

上記の「物体領域検出」と「画像認識」の結果を合わせることで、最終的に「物体検出」を行います。

<!-- ![](https://i.imgur.com/Bq5716P.jpeg) -->
![img23](img_original/img23.jpeg)  

（https://qiita.com/mdo4nt6n/items/68dcda71e90321574a2b から引用）

一連の流れを画像で表すとこのようになります。

#### YOLOv3での改善点

上記に加えて、v3ではいくつかの工夫を追加して精度や速度の改善を行いました。

1. 1/32、1/16、1/8のサイズに分けて物体検出の結果を出力します。複数のサイズに分けることで、さまざまな画像の大きさに対応可能
2. 物体領域検出の際に使用するバウンディングボックスのサイズを調整

以上で、YOLOについての簡単な講義は終わりです。詳細が気になる方は、Qiitaや個人のブログなどで取り上げられたりまとめられているのでそちらをご参照ください。

## 学習データの作成
Duration: 10

### テスト

それでは、早速学習データを作成したいと思いますが、まずは無事にダウンロードできているかテストを行います。
ノートブックに戻って、重みのダウンロードが完了していたら「E.テスト」を実行しましょう。

<!-- ![](https://i.imgur.com/1y96u5X.jpg) -->
![img24](img_original/img24.jpg)

このように画像が出てきたらテストには問題ありません。検出が上手くいっていなくても大丈夫です。おそらくtinyのため精度が低いのでしょう。

<!-- ![](https://i.imgur.com/FYsQcjM.jpg) -->
![img25](img_original/img25.jpg)

ちなみに、tinyではなく通常の重みを使用すると高精度に検出してくれます。

```python
# tiny
# !python detect.py --weights ./data/weight/yolov3-tiny.tf --tiny --image ./data/meme.jpg

# not tiny
!python detect.py --weights ./data/weight/yolov3.tf --image ./data/meme.jpg

from IPython.display import Image,display_jpeg
display_jpeg(Image('output.jpg'))
```
先ほどtinyではない重みもダウンロードした方はこちらを実行すればテストできます。

また、他の画像を使用することもできます。

![img26](img_original/img26.jpeg)

左のフォルダマークをクリックして、`drive` > `My Drive` > `Colab Notebooks` > `yolo-tf2-test` > `data`の順にクリックします。

そして、検出させたい画像を最後の`data`フォルダーの中にアップロードします。そのままドラッグアンドドロップするか赤枠のアップロードから画像を選ぶだけです。

```python=
# tiny
!python detect.py --weights ./data/weight/yolov3-tiny.tf --tiny --image ./data/meme.jpg # ← `meme.jpg` の箇所を検出したい画像ファイル名にする

# not tiny
# !python detect.py --weights ./data/weight/yolov3.tf --image ./data/meme.jpg

from IPython.display import Image,display_jpeg
display_jpeg(Image('output.jpg'))
```
画像をアップロードしたら、2行目の`meme.jpg`のところをアップロードしたファイル名に変更して、再度実行すればOKです。

```python=
# tiny
!python detect.py --weights ./data/weight/yolov3-tiny.tf --tiny --image ./data/test.jpg # ← `meme.jpg` の箇所を検出したい画像ファイル名にする

# not tiny
# !python detect.py --weights ./data/weight/yolov3.tf --image ./data/meme.jpg

from IPython.display import Image,display_jpeg
display_jpeg(Image('output.jpg'))
```

たとえば、`test.jpg`を対象に実行する場合、上記のように変更します。

それでは、学習データを作成してきます。

物体検出の場合、学習データは「画像」と「バウンディングボックス（正確にはバウンディングボックスの座標位置）」となります。

画像は、みなさんが用意したものを使用します。その画像に「VoTT」というツールを使って検出したい箇所にバウディングボックスをつけていきます（この作業をアノテーションと呼びます）。

### オリジナルデータの解凍

まずは、用意した画像を`original_images`という名前のフォルダーにまとめてください。まとめたら`original_images.zip`とzip形式に圧縮してください。

用意する画像のサイズや拡張子はなんでもOKです（後ほど自動でリサイズします。）。拡張子は揃えておいた方がほんの少しだけ楽です。
スマートフォンで撮った写真を使っても大丈夫です。

![img26-1](img_original/img26-1.jpeg)

画像はどういったものでも問題ありませんが、始めはシンプルに検出したい物だけを映すことをオススメします。まずは、シンプルなデータで学習して物体検出を体感してみましょう。

圧縮したらGoogle Driveの`data`フォルダー内にアップロードします。アップロードしたら「F.オリジナルデータの解凍」を実行します。アップロードしたzipフォルダーを解凍します。

![img26-2](img_original/img26-2.jpeg)

画像をさらにフォルダー毎に分ける必要はありません。上の画像のように、画像データだけを`original_images`フォルダーに入れてください。

### データの整形

解凍が終わったら「G.データの整形」を行い、416*416サイズに変更します。サイズ変更した画像は`resized_images`に保存されます。

実行する前に拡張子をプログラム内で指定する必要があります。

```python
extension = 'jpg'
```
8行目の`jpg`の箇所を皆さんのデータの拡張子に揃えてください。

トリミングではなく、1:1となるようにリサイズし足りない箇所は白色で埋めるようにしています。

複数の拡張子が混在している場合は、お手数ですがすべての拡張子に対してこの作業をお願いします。

### 画像の水増し

次に画像の水増しを行います。
```python
############
volume = 60 # 1枚あたりどれだけ増やすか
img_name = 'img' # 生成された画像のファイル名
############
```
この箇所の数値を変更することで、画像1枚に対して水増しする量を調整できます。上記の例の場合、1枚に対しておよそ60枚複製されることになります。

枚数を決定したら「H.画像の水増し」を実行しましょう。最初は1クラスに対して合計100枚ぐらいになるように設定することをオススメします。一度簡単に試してみて上手く検出できていたら画像を増やしていく方が効率的です。

また、新たに画像を追加する場合は、`img_name = 'img'`の`img`部分を変更して、水増しされた画像のファイル名が被らないようにしましょう。（言ってることがわかりにくい方は、実際に水増しされたあとの画像ファイル名を確認してみるとわかりやすいです）

```python
datagen = ImageDataGenerator(
    channel_shift_range=30,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.1,
    )
```
画像の水増しは、ただ複製するだけではなく少し加工した上で複製を行います。どのように加工するか指定するパラメーターについては、[こちら](https://qiita.com/takurooo/items/c06365dd43914c253240)で詳しく解説されていますのでご参照ください。

ここも調整することで精度が変化するので各自色々と試してみましょう。

### 変形した画像を圧縮してダウンロード

データを水増しができたら最後に「I.変形した画像を圧縮してダウンロード」を実行します。実行すると自動でダウンロードされますが、たまに、ポップアップが出現しダウンロードの許可ボタンを押す必要がありますので注意してください。

これで画像を用意できたので、次はアノテーションを行っています。

まずは、VoTTを[こちら](https://github.com/Microsoft/VoTT/releases)からダウンロードしてください。

<!-- ![](https://i.imgur.com/pWx2odP.jpeg) -->
![img27](img_original/img27.jpeg)

自分のOSに合わせて`.dmg`,`.snap`,`.exe`ファイルのどれかをダウンロードしてください。ダウンロードしたらファイルを実行して、VoTTを立ち上げましょう。

<!-- ![](https://i.imgur.com/fN6RNbs.jpeg) -->
![img28](img_original/img28.jpeg)

立ち上がったら`New Project`を選びます。

<!-- ![](https://i.imgur.com/nBW0G9v.jpeg) -->
![img29](img_original/img29.jpeg)

プロジェクトの名前などを設定します。

- Display Name: プロジェクトの名前を記入
- Security Token: とくに設定不要
- Source Connection: アノテーションする画像フォルダーを選ぶ
- Target Connection: データの出力先を選ぶ
- Description: とくに設定不要
- Video Settings: 動画のフレーム数を設定、今回は不要
- Tags: タグの作成

Tagsに検出したい画像の名前を追加してください。

Source / Target Connectionは、まずは`Add Connection`から新しく作成する必要があります。

<!-- ![](https://i.imgur.com/szcHprr.jpeg) -->
![img30](img_original/img30.jpeg)

- Display Name: Connectionimの名前
- Provider: Local File Systemを選ぶ
- Local File System: ローカル内の任意のディレクトリを選ぶ

出力先のフォルダーは、別途`out`のような名前のフォルダーを追加するとわかりやすいです。

<!-- ![](https://i.imgur.com/qnm3aiF.jpeg) -->
![img31](img_original/img31.jpeg)

このようにダウンロードしてきた`changed_images`をSource　Connectionに設定し、その中に`out`を作ってTarget Connectionにするのがお勧めです。

それぞれ設定できたら`Save Connection`をクリックします。これでアノテーションに入れますが、もう少し設定を変更します。

<!-- ![](https://i.imgur.com/TZgGh3S.jpeg) -->
![img32](img_original/img32.jpeg)

まずは、左の上から4つ目のアイコンをクリックして、出力形式を上記画像のように変更します。

- Provider: Pascal VOC
- Asset State: Only tagged Assets
- Test / Train Split: 100

変更したら、`Save Export Settings`をクリックします。

<!-- ![](https://i.imgur.com/PzS3Ley.jpeg) -->
![img33](img_original/img33.jpeg)

次に、左の上から5番目のアイコンをクリックして、上記のように設定します。VoTTの機能でアノテーションを補助してくれます。ここでは、自動でバウンディングボックスをつけるようにしています。不要なバウンディングボックスが多ければ、`Auto Detect`のチェックマークを外してください。

変更したら、`Save Project`をクリックします。

これで設定はお終いです。左の上から2番目をクリックしてアノテーション作業に入ります。

<!-- ![](https://i.imgur.com/JxNBQfN.jpg) -->
![img34](img_original/img34.jpg)

直感的に操作できるUIですが、簡単に操作説明をします。

1. アノテーション用のボックスを作成
2. 前/次の画像に移動
3. 保存
4. データの出力
5. タグの選択

基本的に1 → 5 → 2の流れで行います。全部アノテーションしたら、4をクリックしてください。

その他以下の機能があります。
- タグは鍵マークをクリックすると固定になり自動でそのタグがつく
- タグの右側にある数字キーを押すとそのタグがつく
- バウンディングボックスは重なっていても問題ない
- タグの左側をクリックした後に、鉛筆ボタンをクリックすると色を選べる（どの色を使っても問題ない）
- Wキーを押すと前の画像に移動
- Sキーを押すと次の画像に移動

以上が、VoTTの使い方です。細かな使い方や機能は各自で調べてもらえればと思います。慣れると単調な作業になり辛いですが、根気よく行うしかありません。

## 学習
Duration: 10

次に学習を実行します。画像の量に応じて、「画像の解凍」「tfrecordへの変換」「学習」それぞれに時間がかかります。

### 画像の解凍

アノテーション作業が終わったら、VoTTから必要なデータを出力します。出力ボタンを押すだけで、Target Connectionで設定したフォルダーにデータが出力されます。

<!-- ![](https://i.imgur.com/A8NxHu7.jpeg) -->
![img35](img_original/img35.jpeg)

右上にこのようなポップが表示されたら無事出力成功です。

<!-- ![](https://i.imgur.com/rW8Cq0P.jpeg) -->
![img36](img_original/img36.jpeg)

このようなポップが出てきた場合は、失敗です。原因不明でたまに失敗します。
プロジェクトを保存して、一度VoTTを閉じて再度起動して出力ボタンを再度押してください。また、失敗したらこれを繰り返してください。

<!-- ![](https://i.imgur.com/73l2dON.jpeg) -->
![img37](img_original/img37.jpeg)

Target Connectionには、このようにjsonデータが大量に生成されています。今回必要なのは`~~~~~ -PascalVOC-export`というフォルダーです（変更日など時間でソートをすると見つけやすいです。）。

<!-- ![](https://i.imgur.com/LdrErYw.jpeg) -->
![img38](img_original/img38.jpeg)

このフォルダーを`train_images`に名前を変えて、zip形式に圧縮します。圧縮したら、Google Driveにアップロードします。アップロード先は、同様に`data`フォルダー内です。

アップロードできたら「J.画像の解凍」を実行してください。

### tfrecordの作成

次に用意したデータを「tfrecord」という形式に変換します。tfrecordとは、tensorflowが推奨する軽量なデータフォーマットです。

大量の学習データを1つにまとめていきます。また、合わせてデータの水増しも行っています。

「K.tfrecordの作成」を実行する前に、9~11行目を変更してください。

- `classes_list`: VoTTで使用したタグ名を記載（順番は関係なし）
- `tf_file_name`: tfrecordのファイル名
- `augment_size`: 画像の水増しの量

```python
classes_list = ['rock', 'scisor', 'paper'] # タグ(順番はVoTTと一致していなくても大丈夫)
tf_file_name = 'hand' # ファイル名
augment_size = 1 # 画像の水増し量
```

このような感じで変更してください。
水増し量は、各画像に対して何枚増やすかになります。そのため、数枚程度に留めておいた方が良いと思います。

設定したら「K.tfrecordの作成」を実行してください。

<!-- ![](https://i.imgur.com/6mSVsvk.jpeg) -->
![img39](img_original/img39.jpeg)

実行すると`data`フォルダに`train`,`val`,`test`の3つのtfrecordが生成されます。ファイル名は、`tf_file_name`で指定した文字列 + `_train`,`_val`,`_test`となります。

それぞれ訓練用、検証用、テスト用です。

### tfrecordの表示

tfrecordを作成できたら、テストとして表示させてみます。「L.tfrecordの表示」を実行する前に、14,15行目を設定します。

```python
#######################
tfrecord_name = 'hand' # tfrecordの名前
tfrecord_kind = 'train' # tfrecotdの種類, train, val, test のどれか
#######################
```

`tfrecord_name`は表示させたいtfrecodのファイル名を、`tfrecord_lind`にはtrain,val,testのどれかを設定します。

設定したら「L.tfrecordの表示」を実行しましょう。

バウンディングボックスのついた画像が表示されると思います。

### 学習

これで学習データの準備ができました。次に学習を行っていきます。

「M.学習」を実行する前に、25,26行目を変更してください。

```python
###################################
your_weight_name = 'cola_tea_monster_test'
tfrecord_name = 'cola_tea_monster'
###################################
```
保存する重みのファイル名、学習の際に使用するtfrecordの名前を設定します。設定したら「M.学習」を実行しましょう。画像の量によって学習にかかる時間は変わりますが、100~200枚程度であれば30~60分で終わると思います。

また、ディープラーニングのハイパーパラメーターは、30~34行目で設定できます。

```python
###################################
epochs = 100
batch_size = 16
learning_rate = 1e-3 # 0.001
tiny = True
###################################
```
tinyを`False`にすると通常のYOLOv3の重みで転移学習を行うことができます（もちろん、事前に重みのダウンロードは必要です）。

EaryStoppingを適用しているので、過学習が発生している場合には上記で設定した`epochs`分学習しない場合もあります。

## 物体検出１
Duration: 10

学習している間にラズパイでの物体検出を体感してもらいます。こちらで用意したジャンケンの手を検出する重みを使用します。

まずは、ラズパイを起動してターミナルを起動します。
![img40](img_original/img40.jpeg)

ラズパイのホーム画面の左上の赤枠をクリックするか`Ctrl`+`Alt`+`t`のショートカットキーでも起動できます。

起動したら必要なものをインストールします。パスワードの入力を求められたら、入力してください。

`sudo apt update`

`sudo apt upgrade -y`

`sudo apt install python3-dev python3-pip libatlas-base-dev`

すべて終わったら、`cd Desktop`でデスクトップに移動し、Gitプロジェクトwダウンロードします。

`git clone https://jw-fujibayashi:moyashi314%40tmge@github.com/jw-fujibayashi/yolo-tf2-test.git`

クローンしたら`cd yolo-tf2-test`を実行し、また少し設定を行います。

`wget "https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/master/tensorflow-2.2.0-cp37-cp37m-linux_armv7l_download.sh"`

`chmod 764 ./tensorflow-2.2.0-cp37-cp37m-linux_armv7l_download.sh`

`./tensorflow-2.2.0-cp37-cp37m-linux_armv7l_download.sh`

`sudo pip3 install tensorflow-1.15.0-cp37-cp37m-linux_armv7l.whl`

`sudo pip3 install -r requirements.txt`

無事ダウンロードできたら、[こちら](https://drive.google.com/open?id=1FvNGTYcpmRwvAkKUUwK4YzLDGmlfUPPs)の重みをダウンロードしてください。3ファイルありますが、3ファイルともダウンロードしてください。

ダウンロードできたら、ラズパイの`yolov2-tf-test`の`data`の`weight`フォルダーに入れます。

```shell
python3 detect_video.py --video 0 --tiny --weights data/weight/hands.tf
```

最後に上記コマンドを実行すると、リアルタイム物体検出のプログラムが実行します。ラズパイにカメラをつけた状態で実行する必要があるので注意してください。

プログラムを終了する際には、`q`キーを押してください。

## 物体検出2
Duration: 10

### Google Colabo上でテスト

無事に学習が終わったら、でき上がった重みを使って物体検出を行っていきます。まずは、Goolge Colaboratory上でテストをしてみます。

「N.Google Colabo上でテスト」を実行する前に、16,17行目を変更してください。

```python
###################################
your_weight_name = ''
tfrecord_name = ''
###################################
```

`your_weight_name`にさっきの学習でできた重みファイルの名前を、`tfrecord_name`に使用したtfrecordの名前を設定します（test用のtfrecordを使用します。`_test`は不要です。）。

設定したら「N.Google Colabo上でテスト」を実行してください。上手く検出できたでしょうか？納得のいかない精度の場合は、学習データを増やしたりしてみましょう。（学習のコツは次のスライドでまとめています。）

それでは、最後にラズパイでリアルタイム物体検出をしてみましょう。まずは、作成した「重みファイル3つ（`data`>`weight`フォルダーにあります。）」と「`classes.txt`（`data`フォルダーにあります。）」をダウンロードします。

重みファイルは、`data`の中の`weight`フォルダーに移動させましょう。
`classes.txt`は、`data`フォルダーに移動させます。既にある`classes.txt`は上書きしても大丈夫です。

移動させたらターミナルを開いて、`cd Desktop/yolo-tf2-master`を実行して

```shell
python detect_video.py --video 0 --tiny --weights data/weight/{各自の重みファイル名}.tf
```

を実行します。重みファイルは、`~~~.tf.index`,`~~~.data-~~~`などのようになっていると思いますが、`.tf`までで指定すれば大丈夫です。

## 学習のコツ

自分が試してみて上手くいったと感じたコツを以下にまとめます。論理的な部分もない場合もありますので、軽く参考にしてもらえればと思います。

また、一般的に高精度の物体検出を行うには、1クラスあたり1000枚の画像が必要されると言われています。（単純に水増しして1000枚に達すれば良いわけではありません）

それと比べると今回は圧倒的にデータ量が足りていなく、かつtinyで学習しているので精度に限界があることは念頭に置いておいてください。

### 学習データは量よりも種類

似たような画像をたくさん用意するよりも、バリエーションを増やした方が精度は向上します。たとえば、背景を変えるだけでも大丈夫です。また、遠くから撮った画像なども用意するとより精度は向上します。

その他、角度/光加減などを変えた画像を用意するとなお良くなります。

### 転移学習を活かす

今回、転移学習に使っている重みは「coco」と呼ばれるデータセットを使用して学習された重みです。
- person
- bicycle
- car
- motorbike
- aeroplane
- bus
- train
- truck
- boat
- traffic light
- fire hydrant
- stop sign
- parking meter
- bench
- bird
- cat
- dog
- horse
- sheep
- cow
- elephant
- bear
- zebra
- giraffe
- backpack
- umbrella
- handbag
- tie
- suitcase
- frisbee
- skis
- snowboard
- sports ball
- kite
- baseball bat
- baseball glove
- skateboard
- surfboard
- tennis racket
- bottle
- wine glass
- cup
- fork
- knife
- spoon
- bowl
- banana
- apple
- sandwich
- orange
- broccoli
- carrot
- hot dog
- pizza
- donut
- cake
- chair
- sofa
- pottedplant
- bed
- diningtable
- toilet
- tvmonitor
- laptop
- mouse
- remote
- keyboard
- cell phone
- microwave
- oven
- toaster
- sink
- refrigerator
- book
- clock
- vase
- scissors
- teddy bear
- hair drier
- toothbrush

たとえば、ペットボトルのお茶とコカコーラの物体検出は、比較的少ない画像データで精度が高くなりました。おそらく、bottoleが事前に学習されていたからだと思います。

### 推論時と同じ環境のデータを用意する

どんな時でも物体検出を高精度で行う重みを作成するのは大変です。そのため、AIが推論を上手く行えるように寄り添ってあげる必要があります。

その1つとして、学習データを推論時に映る風景を合わせてあげると精度が上ります。

たとえば、防犯カメラのような定点カメラとして使用するのであれば、その風景でのみ物体検出できれば良いので学習データを合わせてあげると良いと思います。
# 必要なモジュールのインポート
import joblib
from flask import Flask, request, render_template
from wtforms import Form, FloatField, SubmitField, validators
import numpy as np

'''
1. 学習済みモデルをもとに推論する関数
2. 入力フォームの設定を行うクラス
3. URL にアクセスがあった場合の挙動


入力フォームから数値を受け取る→推論→判定結果を結果表示用のHTMLファイル(result.html)におくる
'''


#　学習済みモデルをもとに推論する関数
def predict(x):
    # 学習済みモデル（iris.pkl）を読み込み
    # model = joblib.load('./src/iris.pkl')
    model = joblib.load('src/iris.pkl')
    x = x.reshape(1,-1) # 2 次元ベクトルに変換
    pred_label = model.predict(x) # 推論
    return pred_label # 呼び出し元に返す

# 推論したラベルから花の名前を返す関数
def getName(label): # ラベルに対応する花の名前を返す
    if label == 0:
        return 'Iris Setosa'
    elif label == 1:
        return 'Iris Versicolor'
    elif label == 2:
        return 'Iris Virginica'
    else:
        return 'Error'

# Flaskのインスタンスを作成
app = Flask(__name__)


'''
2. 入力フォームの設定を行うクラス
    ✓ wtforms を使用してクラスを作成
    ✓ 4 つの数値を入力させたいので、form も4つ用意する
'''
# 入力フォームの設定
class IrisForm(Form):
    SepalLength = FloatField('がくの長さ (0cm ~ 10cm)',
                    # フォームに入力する条件を validatiors を使用して配列で定義
                    [validators.InputRequired(), # form に何も入力されていないかをチェック
                    validators.NumberRange(min=0, max=10, message='0〜10の数値を入力してください')]) # 数値の範囲を絞る条件

    SepalWidth  = FloatField('がくの幅 (0cm ~ 5cm)',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=5, message='0〜5の数値を入力してください')])

    PetalLength = FloatField('花弁の長さ (0cm ~ 10cm)',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=10, message='0〜10の数値を入力してください')])

    PetalWidth  = FloatField('花弁の幅 (0cm ~ 5cm)',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=5, message='0〜5の数値を入力してください')])

    # html 側で表示する submit ボタンの設定
    submit = SubmitField('判定')

'''
3. URL にアクセスがあった場合の挙動
入力フォームからのデータを送信するPOSTメソッドがあるため
POSTとGETの両方の処理を設定する
'''

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts(): # 関数名
    # WTForms で構築したフォームをインスタンス化
    irisForm = IrisForm(request.form) # 入力フォームを作成
    # POST メソッドの定義（submit がクリックされたとき）
    if request.method == 'POST':

        # 条件に当てはまらない場合（未入力や範囲外の場合）
        if irisForm.validate() == False:
            return render_template('index.html', forms=irisForm) # 入力フォームの設定を html へ渡す
        # 条件に当てはまる場合の、推論を実行
        else:
            # 入力フォームで受け取った値を float に変換して各変数名で取得
            VarSepalLength = float(request.form['SepalLength'])
            VarSepalWidth  = float(request.form['SepalWidth'])
            VarPetalLength = float(request.form['PetalLength'])
            VarPetalWidth  = float(request.form['PetalWidth'])
            # 入力された値を np.array に変換して推論
            x = np.array([VarSepalLength, VarSepalWidth, VarPetalLength, VarPetalWidth])
            pred = predict(x)
            irisName_ = getName(pred)
            return render_template('result.html', irisName=irisName_)

    # GET 　メソッドの定義（URLに直接アクセスがあった場合）
    elif request.method == 'GET':
        return render_template('index.html', forms=irisForm)

'''
アプリケーションの実行のためにエントリーポイント内で app.run() を記述
    ・エントリーポイント：プログラムが実行される開始点
    ・debug=True：開発している間は、サーバを立ち上げ直さなくてもWebページに反映させることができる
        実際に運用する際はFalseに変更する。
'''
# アプリケーションの実行
if __name__ == '__main__':
    app.run(debug=True)
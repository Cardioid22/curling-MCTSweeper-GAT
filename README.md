# MDigitalCurling3-MCTSweeper

MCTSweeper は、Monte Carlo Tree Search (MCTS) を使用したデジタルカーリングの思考エンジンです。

## 特徴

- **MCTS アルゴリズム** を用いた意思決定
- **バックプロパゲーションの最適化** による精度向上
- **異なる戦略の選択** (テイクアウト、ガード、ボタン狙いなど)
- **キャッシュの利用** による計算の高速化

## ビルド方法

1. リポジトリをクローンします。
   ```sh
   git clone https://github.com/your-repo/MCTSweeper.git
   cd MCTSweeper
   ```
2. サブモジュールをセットアップします。
   ```sh
   git submodule update --init --recursive
   ```
3. 必要なライブラリをインストールします。
   - [Boost](https://www.boost.org/)
   - [CMake](https://cmake.org/)
4. CMake を用いてビルドします。
   ```sh
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . --config Release
   ```
   :warning: CMake が Boost を見つけられない場合は、環境変数 `BOOST_ROOT` に Boost のインストールディレクトリを設定してください。

## 思考エンジンの概要

MCTSweeper は、Monte Carlo Tree Search を利用して最適なショットを決定するエンジンです。以下の手順で動作します。

1. **シミュレーション**: 現在のゲーム状態から複数の候補ショットを生成。
2. **展開 (Expansion)**: 有望なショットを選択し、シミュレーションを進める。
3. **シミュレーション (Rollout)**: ランダムプレイアウトを実行し、結果を評価。
4. **バックプロパゲーション**: 結果をツリーに反映し、UCT (Upper Confidence Bound for Trees) を更新。
5. **最適なショットの選択**: UCT 値が最も高いショットを採用。

## アルゴリズム

MCTSweeper では、以下の戦略を選択できます。

### 1. 重心による判断

敵ストーンの重心位置を計算し、その位置を基準に戦略を決定。

- **相手ストーンの背後に隠れる**
- **積極的に攻撃する**

### 2. ボタン狙い

ハウスの中心を狙い、安全なショットを選択。

### 3. テイクアウト

ハウス内の相手ストーンを取り除くショット。

### 4. 高速ショット (ブレイク・ザ・ボード)

強い力でストーンを投げ、盤面を大きく変える。

## ライセンス

The Unlicense

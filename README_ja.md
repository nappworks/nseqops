# nseqops - 高性能ベクトル演算ライブラリ

このライブラリは、`float64` 値のシーケンスに対して AVX2 および SSE4.2 命令を使用した高性能なベクトル演算を提供します。計算負荷の高いタスク向けに設計されており、高効率な算術および線形代数演算を実現します。

## 特徴

- **AVX2 および SSE4.2 対応**: AVX2 および SSE4.2 命令セットの両方に最適化され、互換性と性能を確保。
- **ベクトル計算**: ベクトル化による高速な数値計算。
- **包括的な API**: シーケンス同士やスカラーとの計算操作をサポート。
- **高精度**: `float64` 型に対応。

## インストール

### 必要要件

- **Nim >= 2.2.0**: [Nim をインストールする](https://nim-lang.org/)
- **nimsimd >= 1.3.1**: `nimsimd` がインストールされていることを確認してください。
- **AVX2 または SSE4.2 対応 CPU**: 使用する CPU が AVX2 または SSE4.2 命令をサポートしている必要があります。

### セットアップ

`nimble` を使用してライブラリをインストール:

```bash
nimble install https://github.com/nappworks/nseqops
```

## 使用方法

### ライブラリのインポート

AVX2 を使用する場合:
```nim
import nseqops/avx2
```

SSE4.2 を使用する場合:
```nim
import nseqops/sse4
```

### サンプル

#### シーケンスの合計
```nim
let data = @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
echo sumSeq(data)  # 出力: 28.0
```

#### 要素ごとの演算
```nim
let a = @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
let b = @[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

# 要素ごとの加算
echo addSeqs(a, b)  # 出力: @[8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]

# ドット積
echo dotSeqs(a, b)  # 出力: 84.0
```

#### スカラー演算
```nim
# 各要素にスカラー値を加算
echo addSeqScalar(a, 3.0)  # 出力: @[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# 各要素をスカラー値で乗算
echo mulSeqScalar(a, 2.0)  # 出力: @[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
```

## API リファレンス

### シーケンス操作

1. `sumSeq(data: seq[float64]): float64`
2. `prodSeq(data: seq[float64]): float64`
3. `addSeqs(a, b: seq[float64]): seq[float64]`
4. `subSeqs(a, b: seq[float64]): seq[float64]`
5. `mulSeqs(a, b: seq[float64]): seq[float64]`
6. `divSeqs(a, b: seq[float64]): seq[float64]`
7. `dotSeqs(a, b: seq[float64]): float64`

### スカラー操作

1. `addSeqScalar(a: seq[float64], b: float64): seq[float64]`
2. `subSeqScalar(a: seq[float64], b: float64): seq[float64]`
3. `mulSeqScalar(a: seq[float64], b: float64): seq[float64]`
4. `divSeqScalar(a: seq[float64], b: float64): seq[float64]`

## ライセンス

このプロジェクトは MIT ライセンスの下でライセンスされています。詳細は `LICENSE` ファイルを参照してください。

## 追加情報

- このライブラリは現在、`float64` 型のみをサポートしています。
- SSE4.2 の場合、計算は 128 ビットレジスタ（`float64x2`）で実行されます。
- AVX2 の場合、計算は 256 ビットレジスタ（`float64x4`）で実行されます。
- AVX-512 への対応は今後のアップデートで予定されています。

### コンパイル指示

- **AVX を使用する場合**
  ```
  nim c -r --passC:"-mavx" hogehoge.nim
  ```

- **SSE4 を使用する場合**
  ```
  nim c -r --passC:"-msse4" hogehoge.nim
  ```

> **注意**: 適切なフラグを含めないとコンパイルエラーが発生します。

---

問題や追加機能の要望がある場合はお知らせください。

---

import warnings
warnings.filterwarnings("ignore")
import sys
import os
from utils import *

"""
ワークフロー: 
入力 ("In") → create_document()関数によって処理され、トピック、スタイルガイド、基準を受け取る
LLM呼び出し1 → generate_outline()として実装され、初期ドキュメントアウトラインを作成
ゲート → check_outline_criteria()として実装され、アウトラインが要件を満たすか検証
パス経路:
LLM呼び出し2 → expand_outline_sections()で各セクションを詳細なコンテンツに展開
LLM呼び出し3 → write_final_document()で最終的な完成したドキュメントを作成
失敗経路 → 基準が満たされない場合はフィードバックと共にNoneを返す
出力 ("Out") → 最終的なドキュメントテキスト

重要な考慮事項:
- 関心の分離
- 入力を構造化し、区切り文字を使用
- 一貫したフォーマットとスペーシングを使用
- 出力を検証するための基準チェック
- 構造化された出力（この例では適用されていない）
"""

# プロンプト用の区切り文字
outline_delimiter = "<outline>"
outline_delimiter_end = "</outline>"

criteria_delimiter = "<criteria>"
criteria_delimiter_end = "</criteria>"

document_delimiter = "<document>"
document_delimiter_end = "</document>"

def generate_outline(topic, style_guide):
    """初期ドキュメントアウトラインを生成する最初のLLM呼び出し"""
    system_message = f"""
    与えられたトピックについて、構造化されたドキュメントアウトラインを作成することがあなたのタスクです。
    アウトラインは提供されたスタイルガイドに従い、メインセクションとサブセクションを含む必要があります。
    
    アウトラインを以下の形式でフォーマットしてください：
    - メインセクションは数字でマーク（1., 2., 3., など）
    - サブセクションは文字でマーク（a., b., c., など）
    - 各セクションの簡潔な説明
    
    アウトラインを{outline_delimiter}と{outline_delimiter_end}タグの間に出力してください。
    """

    prompt = f"""
    トピック: {topic}
    スタイルガイド: {style_guide}
    
    詳細なアウトラインを生成してください。
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    outline = get_chat_completion(messages)
    return outline

def check_outline_criteria(outline, criteria):
    """アウトラインが指定された基準を満たすか検証するゲート関数"""
    system_message = f"""
    アウトラインがすべての指定された基準を満たすか評価することがあなたのタスクです。
    "PASS"または"FAIL"の後に簡潔な説明のみを返す必要があります。
    
    アウトライン（{outline_delimiter}{outline_delimiter_end}で区切られた）を
    基準（{criteria_delimiter}{criteria_delimiter_end}で区切られた）に対してレビューしてください。
    """

    prompt = f"""
    {outline_delimiter}
    {outline}
    {outline_delimiter_end}
    
    {criteria_delimiter}
    {criteria}
    {criteria_delimiter_end}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    result = get_chat_completion(messages)
    return result.strip().startswith("PASS"), result

def expand_outline_sections(outline):
    """承認されたアウトラインを詳細なセクションに展開する2番目のLLM呼び出し"""
    system_message = f"""
    承認されたアウトラインの各セクションを詳細な段落に展開することがあなたのタスクです。
    各セクションについて：
    1. 元の構造を維持
    2. 2-3段落のコンテンツを追加
    3. 関連する例やサポートポイントを含める
    
    展開されたコンテンツを{document_delimiter}と{document_delimiter_end}タグの間に保持してください。
    """

    prompt = f"""
    このアウトラインを詳細なセクションに展開してください：
    
    {outline_delimiter}
    {outline}
    {outline_delimiter_end}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    expanded_sections = get_chat_completion(messages)
    return expanded_sections

def write_final_document(expanded_sections, style_guide):
    """最終的な完成したドキュメントを作成する3番目のLLM呼び出し"""
    system_message = f"""
    展開されたセクションを完成した、一貫性のあるドキュメントに変換することがあなたのタスクです。
    
    1. セクション間のスムーズな移行を追加
    2. 一貫したトーンとスタイルを確保
    3. 導入と結論を含める
    4. 提供されたスタイルガイドに従う
    
    最終的なドキュメントを{document_delimiter}と{document_delimiter_end}タグの間に出力してください。
    """

    prompt = f"""
    スタイルガイド: {style_guide}
    
    これらの展開されたセクションを完成したドキュメントに変換してください：
    
    {document_delimiter}
    {expanded_sections}
    {document_delimiter_end}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    final_document = get_chat_completion(messages)
    return final_document

def create_document(topic, style_guide, criteria):
    """ドキュメント作成ワークフローを調整するメイン関数"""
    # ステップ1: 初期アウトラインを生成
    outline = generate_outline(topic, style_guide)
    print("✓ 初期アウトラインを生成しました")
    
    # ステップ2: アウトラインが基準を満たすかチェック（ゲート）
    passes_criteria, feedback = check_outline_criteria(outline, criteria)
    print(f"→ 基準チェック: {feedback}")
    
    if not passes_criteria:
        print("✗ ドキュメント作成を停止: アウトラインが基準を満たしませんでした")
        return None
    
    # ステップ3: アウトラインをセクションに展開
    expanded_sections = expand_outline_sections(outline)
    print("✓ アウトラインを詳細なセクションに展開しました")
    
    # ステップ4: 最終的なドキュメントを作成
    final_document = write_final_document(expanded_sections, style_guide)
    print("✓ 最終的なドキュメントを完成しました")
    
    return final_document

if __name__ == "__main__":
    # 使用例
    topic = "現代医療における人工知能の影響"
    
    style_guide = """
    - 専門的で学術的なトーンを使用
    - 現実世界の例を含める
    - 段落を簡潔に保つ（3-4文）
    - 能動態を使用
    - 対象読者: 医療専門家
    """
    
    criteria = """
    1. 少なくとも3つのメインセクションが必要
    2. 利点と課題の両方を含む必要がある
    3. 倫理的考慮事項に対処する必要がある
    4. 将来の展望を含む必要がある
    5. セクション間の明確な論理的な流れが必要
    """
    
    final_document = create_document(topic, style_guide, criteria)
    
    if final_document:
        print("\n最終的なドキュメント:")
        print("=" * 50)
        print(final_document)

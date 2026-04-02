#!/usr/bin/env python3
"""
End-to-end demo: MMFN prediction → blockchain proof submission → audit verification

Usage:
    # 检查服务状态
    python3 demo_e2e.py --health

    # 模拟推理并提交存证
    python3 demo_e2e.py --image test.jpg --dataset weibo

    # 验证已有存证
    python3 demo_e2e.py --verify <receipt_id>
"""

import argparse
import json
import sys

from blockchain_bridge import (
    PredictionPayload,
    health_check,
    submit_proof_with_retry,
    verify_audit,
)


def demo_health() -> int:
    """检查 yuanjing-core 服务是否可用"""
    print("🔍 Checking yuanjing-core service health...")
    if health_check():
        print("✅ Service is healthy and reachable")
        return 0
    else:
        print("❌ Service is not reachable. Is yuanjing-core running?")
        print("   Try: cd yuanjing-core && cargo run")
        return 1


def demo_submit(image_path: str, dataset: str, predicted_label: int, confidence: float, model_hash: str) -> int:
    """
    模拟 AI 推理结果并提交到链上

    在真实场景中，predicted_label 和 confidence 应该来自 MMFN 模型推理
    """
    print("📋 Building prediction payload...")
    print(f"   Image: {image_path}")
    print(f"   Dataset: {dataset}")
    print(f"   Predicted Label: {predicted_label}")
    print(f"   Confidence: {confidence}")

    payload = PredictionPayload(
        dataset=dataset,
        image_path=image_path,
        predicted_label=predicted_label,
        confidence=confidence,
        prompt_pool_hash=model_hash,
    )

    print(f"\n📤 Submitting proof to blockchain...")
    print(f"   Payload: {json.dumps(payload.to_api_payload(), indent=2)}")

    try:
        receipt = submit_proof_with_retry(payload, max_retries=3)
        print("\n✅ Proof submitted successfully!")
        print(f"   Receipt: {json.dumps(receipt, indent=2)}")

        # 尝试获取 receipt_id 用于后续验证
        receipt_id = receipt.get("receipt_id") or receipt.get("id")
        if receipt_id:
            print("\n💡 To verify this proof later, run:")
            print(f"   python3 demo_e2e.py --verify {receipt_id}")

        return 0
    except Exception as e:
        print(f"\n❌ Failed to submit proof: {e}")
        return 1


def demo_verify(receipt_id: str) -> int:
    """验证链上存证"""
    print(f"🔍 Verifying proof: {receipt_id}")

    try:
        result = verify_audit(receipt_id)
        print("\n✅ Verification result:")
        print(f"   {json.dumps(result, indent=2)}")
        return 0
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end demo for MMFN + yuanjing-core integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check service health
    python3 demo_e2e.py --health

    # Submit a fake news detection result (label=0 means fake for weibo)
    python3 demo_e2e.py --image news.jpg --dataset weibo --label 0 --confidence 0.92

    # Submit a real news detection result (label=1 means real for weibo)
    python3 demo_e2e.py --image news.jpg --dataset weibo --label 1 --confidence 0.88

    # Verify an existing proof
    python3 demo_e2e.py --verify abc123
        """,
    )

    parser.add_argument("--health", action="store_true", help="Check if yuanjing-core service is healthy")
    parser.add_argument("--image", type=str, help="Path to the image being analyzed")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["weibo", "gossip"],
        default="weibo",
        help="Dataset name (weibo or gossip)",
    )
    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        default=1,
        help="Predicted label (0=fake, 1=real for weibo)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Prediction confidence (0.0-1.0)",
    )
    parser.add_argument(
        "--model-hash",
        type=str,
        default="0" * 64,
        help="Model version hash (prompt_pool_hash), 64 hex characters",
    )
    parser.add_argument(
        "--verify",
        type=str,
        metavar="RECEIPT_ID",
        help="Verify an existing proof by receipt ID",
    )

    args = parser.parse_args()

    # 检查服务健康状态
    if args.health:
        sys.exit(demo_health())

    # 验证已有存证
    if args.verify:
        sys.exit(demo_verify(args.verify))

    # 提交新存证
    if args.image:
        sys.exit(demo_submit(args.image, args.dataset, args.label, args.confidence, args.model_hash))

    # 没有提供任何参数，显示帮助
    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()

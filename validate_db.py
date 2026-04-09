"""
SMARTER.AI - Database Validation Script
Run this after applying schema.sql to verify everything works.

Usage:
    source ntd/bin/activate
    python validate_db.py
"""

import sys
import time
from database import DatabaseManager


def main():
    print("=" * 60)
    print("SMARTER.AI Database Validation")
    print("=" * 60)

    # Connect
    print("\n[1/8] Connecting to database...")
    try:
        db = DatabaseManager(min_conn=2, max_conn=5)
        print("  OK - Connection pool created")
    except Exception as e:
        print(f"  FAIL - {e}")
        print("\n  Make sure PostgreSQL is running and schema.sql has been applied:")
        print("    psql -U smarter_user -d smarter_ai -h localhost -f schema.sql")
        sys.exit(1)

    try:
        # Create users
        print("\n[2/8] Creating users...")
        users = []
        for i in range(5):
            user = db.get_or_create_user(f"test_user_{i}")
            users.append(user)
            print(f"  Created user {i+1}: {user['user_id']}")
        print(f"  OK - {len(users)} users created")

        # Verify returning user detection
        print("\n[3/8] Testing returning user detection...")
        same_user = db.get_or_create_user("test_user_0")
        assert same_user["user_id"] == users[0]["user_id"], "Returning user not detected!"
        print(f"  OK - Returning user correctly identified: {same_user['user_id']}")

        # Create sessions
        print("\n[4/8] Creating sessions...")
        sessions = []
        professions = ["data scientist", "ML engineer", "researcher"]
        topics = ["neural networks", "NLP", "reinforcement learning", "computer vision", "cybersecurity"]
        for i in range(15):
            user = users[i % len(users)]
            session_id = db.create_session(
                user_id=str(user["user_id"]),
                topic=topics[i % len(topics)],
                profession=professions[i % len(professions)],
                hours_budget=float(2 + (i % 4)),
                learning_format=["slides", "video", "both"][i % 3],
            )
            sessions.append(session_id)
        print(f"  OK - {len(sessions)} sessions created")

        # Update session state and store modules
        print("\n[5/8] Testing session operations...")
        db.update_session_state(sessions[0], "plan_presented")
        db.store_recommended_modules(sessions[0], [
            {"module_idx": 0, "pdf_name": "attention_mechanisms.pdf", "duration": 45},
            {"module_idx": 1, "pdf_name": "transformers_intro.pdf", "duration": 30},
        ])
        session = db.get_session(sessions[0])
        assert session["session_state"] == "plan_presented", "Session state not updated!"
        assert len(session["recommended_modules"]) == 2, "Modules not stored!"
        user_sessions = db.get_user_sessions(str(users[0]["user_id"]))
        assert len(user_sessions) >= 1, "User sessions not retrieved!"
        print(f"  OK - Session state updated, modules stored, {len(user_sessions)} sessions retrieved")

        # Save messages
        print("\n[6/8] Saving messages...")
        msg_count = 0
        sample_messages = [
            ("user", "I want to learn about attention mechanisms", "general"),
            ("assistant", "Great topic! Let me find relevant modules for you.", "general"),
            ("user", "Explain the kernel trick from module 1", "module_qa"),
            ("assistant", "The kernel trick maps data to higher dimensions...", "module_qa"),
            ("user", "My fraud detection model isn't generalising", "professional_problem"),
        ]
        for sid in sessions[:10]:
            for role, content, msg_type in sample_messages:
                module_ref = {"module_idx": 0, "pdf_name": "attention_mechanisms.pdf"} if msg_type == "module_qa" else None
                db.save_message(sid, role, content, msg_type, module_ref)
                msg_count += 1
        # Add extra messages to reach ~200
        for sid in sessions:
            for j in range(8):
                db.save_message(sid, "user", f"Test message {j} in session", "general")
                db.save_message(sid, "assistant", f"Response {j} in session", "general")
                msg_count += 2
        print(f"  OK - {msg_count} messages saved")

        # Test message retrieval
        msgs = db.get_session_messages(sessions[0], limit=5)
        assert len(msgs) <= 5, "Message limit not respected!"
        assert msgs[0]["created_at"] <= msgs[-1]["created_at"], "Messages not in chronological order!"
        print(f"  OK - Retrieved {len(msgs)} messages in chronological order")

        # Module interactions
        print("\n[7/8] Testing module interactions...")
        interaction_count = 0
        for sid in sessions[:10]:
            for mod_idx in range(3):
                iid = db.create_or_update_module_interaction(
                    sid, mod_idx, f"module_{mod_idx}.pdf"
                )
                db.increment_question_count(iid)
                db.increment_question_count(iid)
                db.add_concept(iid, "backpropagation")
                db.add_concept(iid, "gradient descent")
                db.add_concept(iid, "backpropagation")  # duplicate — should NOT be added
                db.add_professional_app(iid, "Applied to fraud detection model")
                interaction_count += 1

        # Verify duplicate concept was not added
        test_interaction = db.get_module_interaction(iid)
        concepts = test_interaction["concepts_asked"]
        assert concepts.count("backpropagation") == 1, f"Duplicate concept added! concepts={concepts}"
        assert test_interaction["question_count"] == 2, "Question count wrong!"
        print(f"  OK - {interaction_count} interactions created")
        print(f"  OK - Duplicate concept correctly rejected (concepts: {concepts})")
        print(f"  OK - Question count correct: {test_interaction['question_count']}")

        # Cross-session query
        user_interactions = db.get_user_module_interactions(str(users[0]["user_id"]))
        print(f"  OK - Cross-session query returned {len(user_interactions)} interactions for user 0")

        # Professional problems
        print("\n[8/8] Testing professional problems...")
        for sid in sessions[:5]:
            db.create_professional_problem(
                sid,
                "My classification model has high bias",
                related_modules=[{"module_idx": 0}],
                related_concepts=["bias-variance tradeoff"],
            )
        problems = db.get_session_problems(sessions[0])
        assert len(problems) >= 1, "Problems not retrieved!"
        user_problems = db.get_user_problems(str(users[0]["user_id"]))
        print(f"  OK - {len(problems)} session problems, {len(user_problems)} user problems")

        # Summary
        print("\n" + "=" * 60)
        print("ALL VALIDATIONS PASSED")
        print("=" * 60)
        print(f"\n  Users:              5")
        print(f"  Sessions:           {len(sessions)}")
        print(f"  Messages:           {msg_count}")
        print(f"  Module interactions: {interaction_count}")
        print(f"  Professional problems: 5")
        print(f"\n  Database is ready for Phase 2.")

    except AssertionError as e:
        print(f"\n  ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
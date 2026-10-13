"""Deterministic generation script for InferPath Pilot Evaluation Dataset (320 prompts).

Generates 80 distinct task families (4 items each) across:
- 8 required task categories (10 families per category).
- 3 stratification tiers (easy, moderate, difficult).
- 2 leakage-free splits (dev: 40 families / 160 items, test: 40 families / 160 items).
- Realistic prompts, paraphrases, exact repeats, near-duplicates, time-sensitive, and private items.
"""

import json
from pathlib import Path
from typing import Any

from inferpath.models.enums import PrivacyClass


def build_dataset() -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    item_counter = 1

    # 1. Arithmetic & Unit Conversion (Tier: Easy / Moderate)
    # -------------------------------------------------------------
    math_families = [
        # Dev families
        {
            "fam": "fam_math_arith_01",
            "tier": "easy",
            "split": "dev",
            "cat": "arithmetic_and_unit_conversion",
            "items": [
                {
                    "prompt": "Calculate: 125 * 8 + 450",
                    "checker_type": "exact",
                    "cfg": {"reference": "1450"},
                    "criteria": ["Accurate calculation result"],
                },
                {
                    "prompt": "What is 125 multiplied by 8 plus 450?",
                    "checker_type": "exact",
                    "cfg": {"reference": "1450"},
                    "criteria": ["Accurate calculation result"],
                },  # Paraphrase
                {
                    "prompt": "Calculate: 125 * 8 + 450",
                    "checker_type": "exact",
                    "cfg": {"reference": "1450"},
                    "criteria": ["Accurate calculation result"],
                },  # Exact repeat
                {
                    "prompt": "Calculate: 125 * 8 + 450 and output JSON",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"1450"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format", "Value 1450"],
                },  # Near-duplicate with JSON constraint
            ],
        },
        {
            "fam": "fam_math_temp_c2f_01",
            "tier": "easy",
            "split": "dev",
            "cat": "arithmetic_and_unit_conversion",
            "items": [
                {
                    "prompt": "Convert 25°C to Fahrenheit",
                    "checker_type": "exact",
                    "cfg": {"reference": "77°F"},
                    "criteria": ["Exact temperature 77°F"],
                },
                {
                    "prompt": "Convert 25 degrees Celsius into Fahrenheit",
                    "checker_type": "exact",
                    "cfg": {"reference": "77°F"},
                    "criteria": ["Exact temperature 77°F"],
                },  # Paraphrase
                {
                    "prompt": "Convert 25°C to Fahrenheit",
                    "checker_type": "exact",
                    "cfg": {"reference": "77°F"},
                    "criteria": ["Exact temperature 77°F"],
                },  # Exact repeat
                {
                    "prompt": "Convert 25°C to Fahrenheit for an engineering report",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"77(?:\.0)?\s*°?F"},
                    "params": {"audience": "engineering"},
                    "criteria": ["Exact temperature 77°F"],
                },
            ],
        },
        {
            "fam": "fam_math_speed_01",
            "tier": "easy",
            "split": "dev",
            "cat": "arithmetic_and_unit_conversion",
            "items": [
                {
                    "prompt": "Convert 100 km/h to mph",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"62\.14"},
                    "criteria": ["Accurate speed conversion (~62.14 mph)"],
                },
                {
                    "prompt": "How many miles per hour is 100 kilometers per hour?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"62\.14"},
                    "criteria": ["Accurate speed conversion (~62.14 mph)"],
                },
                {
                    "prompt": "Convert 100 km/h to mph",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"62\.14"},
                    "criteria": ["Accurate speed conversion (~62.14 mph)"],
                },  # Repeat
                {
                    "prompt": "Convert 100 km/h to mph in JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"62\.14"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON output", "Value 62.14"],
                },
            ],
        },
        {
            "fam": "fam_math_pct_01",
            "tier": "easy",
            "split": "dev",
            "cat": "arithmetic_and_unit_conversion",
            "items": [
                {
                    "prompt": "Calculate a 15% tip on a bill of $84.00",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 12.60, "tolerance": 0.05},
                    "criteria": ["Tip value $12.60"],
                },
                {
                    "prompt": "What is 15 percent of 84 dollars?",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 12.60, "tolerance": 0.05},
                    "criteria": ["Tip value $12.60"],
                },
                {
                    "prompt": "Calculate a 15% tip on a bill of $84.00",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 12.60, "tolerance": 0.05},
                    "criteria": ["Tip value $12.60"],
                },  # Repeat
                {
                    "prompt": "Calculate a 15% tip on $84.00 and total amount",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"96\.60"},
                    "criteria": ["Total $96.60"],
                },
            ],
        },
        {
            "fam": "fam_math_time_sens_01",
            "tier": "moderate",
            "split": "dev",
            "cat": "arithmetic_and_unit_conversion",
            "items": [
                {
                    "prompt": "What is the current live exchange rate from USD to EUR today?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["rate", "EUR", "USD"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Fresh exchange rate", "Refusal of stale cache"],
                },
                {
                    "prompt": "Current USD to EUR currency exchange rate right now today",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["rate", "EUR"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Fresh exchange rate"],
                },
                {
                    "prompt": "What is the live exchange rate for USD to Japanese Yen right now?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Yen", "USD"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Fresh rate"],
                },
                {
                    "prompt": "Today's live spot price of Bitcoin in USD right now",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Bitcoin", "USD"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Fresh crypto price"],
                },
            ],
        },
        # Test families
        {
            "fam": "fam_math_temp_f2c_02",
            "tier": "easy",
            "split": "test",
            "cat": "arithmetic_and_unit_conversion",
            "items": [
                {
                    "prompt": "Convert 68°F to Celsius",
                    "checker_type": "exact",
                    "cfg": {"reference": "20°C"},
                    "criteria": ["Exact 20°C"],
                },
                {
                    "prompt": "What is 68 degrees Fahrenheit in Celsius?",
                    "checker_type": "exact",
                    "cfg": {"reference": "20°C"},
                    "criteria": ["Exact 20°C"],
                },
                {
                    "prompt": "Convert 68°F to Celsius",
                    "checker_type": "exact",
                    "cfg": {"reference": "20°C"},
                    "criteria": ["Exact 20°C"],
                },  # Repeat
                {
                    "prompt": "Convert 68°F to Celsius as a concise JSON object",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"20(?:\.0)?\s*°?C"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format", "Value 20°C"],
                },
            ],
        },
        {
            "fam": "fam_math_len_m2ft_02",
            "tier": "easy",
            "split": "test",
            "cat": "arithmetic_and_unit_conversion",
            "items": [
                {
                    "prompt": "Convert 10 meters to feet",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"32\.8[0-9]*"},
                    "criteria": ["Accurate conversion ~32.81 feet"],
                },
                {
                    "prompt": "How many feet are in 10 meters?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"32\.8[0-9]*"},
                    "criteria": ["Accurate conversion ~32.81 feet"],
                },
                {
                    "prompt": "Convert 10 meters to feet",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"32\.8[0-9]*"},
                    "criteria": ["Accurate conversion ~32.81 feet"],
                },  # Repeat
                {
                    "prompt": "Convert 10 meters to feet in Spanish",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"32\.8[0-9]*"},
                    "params": {"language": "es"},
                    "criteria": ["Spanish language", "Value ~32.81"],
                },
            ],
        },
        {
            "fam": "fam_math_wt_kg2lb_02",
            "tier": "easy",
            "split": "test",
            "cat": "arithmetic_and_unit_conversion",
            "items": [
                {
                    "prompt": "Convert 75 kg to pounds",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"165\.3[0-9]*"},
                    "criteria": ["Accurate conversion ~165.35 lbs"],
                },
                {
                    "prompt": "How many pounds in 75 kilograms?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"165\.3[0-9]*"},
                    "criteria": ["Accurate conversion ~165.35 lbs"],
                },
                {
                    "prompt": "Convert 75 kg to pounds",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"165\.3[0-9]*"},
                    "criteria": ["Accurate conversion ~165.35 lbs"],
                },  # Repeat
                {
                    "prompt": "Convert 75 kg to pounds for medical records",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"165\.3[0-9]*"},
                    "params": {"audience": "medical"},
                    "criteria": ["Accurate conversion ~165.35 lbs"],
                },
            ],
        },
        {
            "fam": "fam_math_vol_l2gal_02",
            "tier": "easy",
            "split": "test",
            "cat": "arithmetic_and_unit_conversion",
            "items": [
                {
                    "prompt": "Convert 50 liters to gallons",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"13\.2[0-9]*"},
                    "criteria": ["Accurate conversion ~13.2 gallons"],
                },
                {
                    "prompt": "How many US gallons is 50 liters?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"13\.2[0-9]*"},
                    "criteria": ["Accurate conversion ~13.2 gallons"],
                },
                {
                    "prompt": "Convert 50 liters to gallons",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"13\.2[0-9]*"},
                    "criteria": ["Accurate conversion ~13.2 gallons"],
                },  # Repeat
                {
                    "prompt": "Convert 50 liters to gallons in bullet format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"13\.2[0-9]*"},
                    "params": {"expected_format": "bullets"},
                    "criteria": ["Bullet format", "Value 13.2"],
                },
            ],
        },
        {
            "fam": "fam_math_arith_complex_02",
            "tier": "moderate",
            "split": "test",
            "cat": "arithmetic_and_unit_conversion",
            "items": [
                {
                    "prompt": "Calculate: (450 * 12) / 6 + 1500",
                    "checker_type": "exact",
                    "cfg": {"reference": "2400"},
                    "criteria": ["Accurate result 2400"],
                },
                {
                    "prompt": "Evaluate expression: (450 * 12) / 6 + 1500",
                    "checker_type": "exact",
                    "cfg": {"reference": "2400"},
                    "criteria": ["Accurate result 2400"],
                },
                {
                    "prompt": "Calculate: (450 * 12) / 6 + 1500",
                    "checker_type": "exact",
                    "cfg": {"reference": "2400"},
                    "criteria": ["Accurate result 2400"],
                },  # Repeat
                {
                    "prompt": "Calculate: (450 * 12) / 6 + 1500 and show formula step",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"2400"},
                    "criteria": ["Contains 2400"],
                },
            ],
        },
    ]

    # Helper function to generate categories systematically
    def add_family_group(category_name, dev_fams_data, test_fams_data):
        nonlocal item_counter
        for group, split_name in [(dev_fams_data, "dev"), (test_fams_data, "test")]:
            for fam_data in group:
                fam_id = fam_data["fam"]
                tier = fam_data["tier"]
                for itm in fam_data["items"]:
                    priv = itm.get("privacy", PrivacyClass.PUBLIC)
                    is_time = itm.get("time_sensitive", False)
                    cache_elig = (
                        False
                        if (priv != PrivacyClass.PUBLIC or is_time)
                        else itm.get("cache_eligibility", True)
                    )

                    item_dict = {
                        "item_id": f"item_{item_counter:04d}",
                        "prompt": itm["prompt"],
                        "tier": tier,
                        "task_category": category_name,
                        "checker_type": itm.get("checker_type", "rubric"),
                        "checker_config": itm.get("cfg", {}),
                        "split": split_name,
                        "task_family_id": fam_id,
                        "task_parameters": itm.get("params", {}),
                        "privacy_class": priv,
                        "is_time_sensitive": is_time,
                        "cache_eligibility": cache_elig,
                        "quality_criteria": itm.get(
                            "criteria", ["Complete and clear response", "No hallucinations"]
                        ),
                    }
                    items.append(item_dict)
                    item_counter += 1

    # Add Math families
    for fam_def in math_families:
        fam_id = fam_def["fam"]
        split_name = fam_def["split"]
        tier = fam_def["tier"]
        cat = fam_def["cat"]
        for itm in fam_def["items"]:
            priv = itm.get("privacy", PrivacyClass.PUBLIC)
            is_time = itm.get("time_sensitive", False)
            cache_elig = (
                False
                if (priv != PrivacyClass.PUBLIC or is_time)
                else itm.get("cache_eligibility", True)
            )

            item_dict = {
                "item_id": f"item_{item_counter:04d}",
                "prompt": itm["prompt"],
                "tier": tier,
                "task_category": cat,
                "checker_type": itm["checker_type"],
                "checker_config": itm["cfg"],
                "split": split_name,
                "task_family_id": fam_id,
                "task_parameters": itm.get("params", {}),
                "privacy_class": priv,
                "is_time_sensitive": is_time,
                "cache_eligibility": cache_elig,
                "quality_criteria": itm["criteria"],
            }
            items.append(item_dict)
            item_counter += 1

    # 2. Templated Documents
    dev_tpl = [
        {
            "fam": "fam_tpl_leave_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Draft a leave application for John Doe from 2026-11-01 to 2026-11-05 for family emergency",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"John Doe.*2026-11-01.*family emergency"},
                    "criteria": ["Contains name", "Dates", "Reason"],
                },
                {
                    "prompt": "Please write a formal leave request for John Doe for 2026-11-01 to 2026-11-05 due to family emergency",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"John Doe.*2026-11-01"},
                    "criteria": ["Formal tone", "Dates", "Name"],
                },
                {
                    "prompt": "Draft a leave application for John Doe from 2026-11-01 to 2026-11-05 for family emergency",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"John Doe"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Draft a leave application for Alice without specifying start date or reason",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["clarification", "date", "reason"]},
                    "criteria": ["Requests missing date and reason clarification"],
                },  # Underspecified
            ],
        },
        {
            "fam": "fam_tpl_meeting_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Draft meeting minutes for Sprint Retro attended by Alice, Bob with decision to adopt ruff",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Sprint Retro.*Alice.*Bob.*ruff"},
                    "criteria": ["Meeting title", "Attendees", "Decisions"],
                },
                {
                    "prompt": "Record meeting notes for Sprint Retro with participants Alice, Bob deciding to use ruff",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Sprint Retro.*Alice"},
                    "criteria": ["Structured minutes"],
                },
                {
                    "prompt": "Draft meeting minutes for Sprint Retro attended by Alice, Bob with decision to adopt ruff",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Sprint Retro"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Draft meeting minutes for Sprint Retro with attendees missing agenda and decisions",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["agenda", "decision"]},
                    "criteria": ["Detects missing items"],
                },
            ],
        },
        {
            "fam": "fam_tpl_email_client_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Draft a formal email to client Acme Corp regarding project kickoff on Monday",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Acme Corp.*project kickoff"},
                    "criteria": ["Client name", "Subject", "Professional greeting"],
                },
                {
                    "prompt": "Write a professional kickoff notification email to Acme Corp for Monday launch",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Acme Corp"},
                    "criteria": ["Formal email structure"],
                },
                {
                    "prompt": "Draft a formal email to client Acme Corp regarding project kickoff on Monday",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Acme Corp"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Draft formal email to Acme Corp regarding project kickoff in French",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Acme", "projet"]},
                    "params": {"language": "fr"},
                    "criteria": ["French language"],
                },
            ],
        },
        {
            "fam": "fam_tpl_support_ack_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Draft customer support ticket acknowledgement for ticket #4082 regarding billing dispute",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"#?4082.*billing"},
                    "criteria": ["Ticket reference #4082", "Polite acknowledgement"],
                },
                {
                    "prompt": "Write a support receipt email acknowledging ticket 4082 on billing issue",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"4082"},
                    "criteria": ["Ticket receipt"],
                },
                {
                    "prompt": "Draft customer support ticket acknowledgement for ticket #4082 regarding billing dispute",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"4082"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Draft customer support ticket acknowledgement for ticket #4082 in concise JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"4082"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format"],
                },
            ],
        },
        {
            "fam": "fam_tpl_private_nda_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Draft internal mutual NDA agreement for employee Sarah Connor regarding Project Aegis",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["NDA", "Confidentiality", "Aegis"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Strict confidentiality", "Internal only"],
                },
                {
                    "prompt": "Prepare internal NDA document template for Sarah Connor on Project Aegis",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["NDA", "Aegis"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["NDA template"],
                },
                {
                    "prompt": "Draft internal mutual NDA agreement for employee Sarah Connor regarding Project Aegis",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Aegis"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Repeat private"],
                },
                {
                    "prompt": "Draft internal NDA summary for Project Aegis with strict 100 word limit",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Aegis"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "params": {"max_tokens": 120},
                    "criteria": ["Word limit"],
                },
            ],
        },
    ]

    test_tpl = [
        {
            "fam": "fam_tpl_bug_report_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Draft a software bug report for Crash on Login in module auth_v2 with steps to reproduce",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Crash on Login.*auth_v2"},
                    "criteria": ["Bug summary", "Steps to repro", "Module name"],
                },
                {
                    "prompt": "Create an issue report for auth_v2 login crash with expected and actual behavior",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"auth_v2"},
                    "criteria": ["Issue structure"],
                },
                {
                    "prompt": "Draft a software bug report for Crash on Login in module auth_v2 with steps to reproduce",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"auth_v2"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Draft a software bug report for Crash on Login in markdown with headings # Summary, # Repro",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"#\s*Summary.*#\s*Repro"},
                    "params": {"expected_format": "markdown"},
                    "criteria": ["Markdown headings"],
                },
            ],
        },
        {
            "fam": "fam_tpl_invoice_cover_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Draft an invoice cover letter for Invoice #9921 totaling $4,500 due on 2026-12-01 to client Globex",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"9921.*4,?500.*Globex"},
                    "criteria": ["Invoice number", "Amount", "Client Globex"],
                },
                {
                    "prompt": "Write invoice transmittal letter for Globex on invoice 9921 for $4,500",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Globex.*9921"},
                    "criteria": ["Transmittal letter"],
                },
                {
                    "prompt": "Draft an invoice cover letter for Invoice #9921 totaling $4,500 due on 2026-12-01 to client Globex",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"9921"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Draft invoice cover letter for Invoice #9921 to Globex in German",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Globex", "Rechnung"]},
                    "params": {"language": "de"},
                    "criteria": ["German language"],
                },
            ],
        },
        {
            "fam": "fam_tpl_performance_rev_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Draft an employee annual performance review template for Senior Engineer role with goals and competencies",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Goals", "Competencies", "Performance"]},
                    "criteria": ["Review structure", "Competencies"],
                },
                {
                    "prompt": "Create an evaluation form for senior engineer annual performance appraisal",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["evaluation", "performance"]},
                    "criteria": ["Appraisal form"],
                },
                {
                    "prompt": "Draft an employee annual performance review template for Senior Engineer role with goals and competencies",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Senior Engineer"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Draft annual performance review template for Senior Engineer in strict JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{.*\"goals\".*\}"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format"],
                },
            ],
        },
        {
            "fam": "fam_tpl_offer_letter_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Draft a formal job offer letter for Bob Vance as Data Analyst at Initech starting 2026-11-15",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Bob Vance.*Data Analyst.*Initech"},
                    "criteria": ["Candidate Bob Vance", "Role Data Analyst", "Company Initech"],
                },
                {
                    "prompt": "Generate an employment offer letter for candidate Bob Vance for Initech Data Analyst position",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Bob Vance.*Initech"},
                    "criteria": ["Offer letter structure"],
                },
                {
                    "prompt": "Draft a formal job offer letter for Bob Vance as Data Analyst at Initech starting 2026-11-15",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Bob Vance"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Draft job offer letter for Bob Vance with compensation details left underspecified",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["salary", "compensation"]},
                    "criteria": ["Notes missing compensation"],
                },
            ],
        },
        {
            "fam": "fam_tpl_internal_security_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Draft internal incident response playbook notification for critical database breach in cluster DB-03",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Incident", "DB-03", "breach"]},
                    "privacy": PrivacyClass.INTERNAL,
                    "criteria": ["Internal security policy", "Cluster DB-03"],
                },
                {
                    "prompt": "Prepare internal security alert memo on DB-03 security incident",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["DB-03", "incident"]},
                    "privacy": PrivacyClass.INTERNAL,
                    "criteria": ["Security alert"],
                },
                {
                    "prompt": "Draft internal incident response playbook notification for critical database breach in cluster DB-03",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["DB-03"]},
                    "privacy": PrivacyClass.INTERNAL,
                    "criteria": ["Repeat internal"],
                },
                {
                    "prompt": "Draft internal DB-03 breach alert in JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"DB-03"},
                    "privacy": PrivacyClass.INTERNAL,
                    "params": {"expected_format": "json"},
                    "criteria": ["Internal JSON"],
                },
            ],
        },
    ]
    add_family_group("templated_documents", dev_tpl, test_tpl)

    # 3. Rewriting (40 items, 10 families)
    dev_rew = [
        {
            "fam": "fam_rew_formal_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Rewrite in formal tone: 'hey guys, we gotta push back the deadline cuz the build is totally broken.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["deadline", "postpone", "build"]},
                    "criteria": ["Formal business tone", "Accurate meaning"],
                },
                {
                    "prompt": "Make this professional: 'hey guys, we gotta push back the deadline cuz the build is totally broken.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["deadline", "delay"]},
                    "criteria": ["Professional phrasing"],
                },
                {
                    "prompt": "Rewrite in formal tone: 'hey guys, we gotta push back the deadline cuz the build is totally broken.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["deadline"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Rewrite in formal tone: 'hey guys, we gotta push back the deadline' in under 15 words",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["deadline"]},
                    "params": {"max_tokens": 20},
                    "criteria": ["Length constraint"],
                },
            ],
        },
        {
            "fam": "fam_rew_active_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Convert from passive to active voice: 'The quarterly financial report was reviewed by the audit committee.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Tt]he audit committee reviewed"},
                    "criteria": ["Active subject: The audit committee reviewed"],
                },
                {
                    "prompt": "Rewrite using active voice: 'The quarterly financial report was reviewed by the audit committee.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Tt]he audit committee reviewed"},
                    "criteria": ["Active subject"],
                },
                {
                    "prompt": "Convert from passive to active voice: 'The quarterly financial report was reviewed by the audit committee.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Tt]he audit committee reviewed"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Rewrite into active voice in Spanish: 'The report was reviewed by the audit committee.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["comité"]},
                    "params": {"language": "es"},
                    "criteria": ["Spanish active voice"],
                },
            ],
        },
        {
            "fam": "fam_rew_bullets_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Convert this paragraph into 3 concise bullet points: 'We need to migrate our database to Postgres. Next we must update API routes to use async queries. Finally we should set up connection pooling.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"-\s*.*Postgres.*\n-\s*.*API.*\n-\s*.*pooling"},
                    "params": {"expected_format": "bullets"},
                    "criteria": ["Three bullet points"],
                },
                {
                    "prompt": "Format the migration steps as 3 bullets: database migration to Postgres, update API routes, set up pooling",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"-\s*.*Postgres"},
                    "params": {"expected_format": "bullets"},
                    "criteria": ["Bullet formatting"],
                },
                {
                    "prompt": "Convert this paragraph into 3 concise bullet points: 'We need to migrate our database to Postgres. Next we must update API routes to use async queries. Finally we should set up connection pooling.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Postgres"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Convert the migration steps into numbered steps (1, 2, 3)",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"1\..*2\..*3\."},
                    "params": {"expected_format": "numbered"},
                    "criteria": ["Numbered steps"],
                },
            ],
        },
        {
            "fam": "fam_rew_json_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Extract person profile into valid JSON: 'Carlos Smith is a 34 year old DevOps engineer living in Seattle.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"name\"\s*:\s*\"Carlos Smith\".*\"age\"\s*:\s*34"},
                    "params": {"expected_format": "json"},
                    "criteria": ["Valid JSON", "Keys: name, age, role, city"],
                },
                {
                    "prompt": "Parse into JSON: Carlos Smith, age 34, DevOps engineer, Seattle",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Carlos Smith"},
                    "params": {"expected_format": "json"},
                    "criteria": ["Valid JSON"],
                },
                {
                    "prompt": "Extract person profile into valid JSON: 'Carlos Smith is a 34 year old DevOps engineer living in Seattle.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Carlos Smith"},
                    "params": {"expected_format": "json"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Extract person profile into YAML format: Carlos Smith, age 34, DevOps, Seattle",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"name:\s*Carlos Smith"},
                    "params": {"expected_format": "yaml"},
                    "criteria": ["YAML format"],
                },
            ],
        },
        {
            "fam": "fam_rew_confidential_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Clean and rephrase internal system credentials architecture memo: 'Database master key is stored in Vault cluster K-9 under namespace prod-alpha.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Vault", "cluster", "namespace"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Preserves privacy", "Clear phrasing"],
                },
                {
                    "prompt": "Rephrase internal architecture memo regarding Vault K-9 cluster credentials",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Vault"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Confidentiality"],
                },
                {
                    "prompt": "Clean and rephrase internal system credentials architecture memo: 'Database master key is stored in Vault cluster K-9 under namespace prod-alpha.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Vault"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Repeat confidential"],
                },
                {
                    "prompt": "Summarize Vault cluster memo in 1 sentence for internal security team",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Vault"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["1 sentence internal summary"],
                },
            ],
        },
    ]

    test_rew = [
        {
            "fam": "fam_rew_concise_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Make this text concise: 'At this point in time, it is of utmost importance that all employees take into consideration the necessity of completing their timesheets prior to the close of business on Friday.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["timesheets", "Friday"]},
                    "criteria": ["Concise (under 15 words)", "Clear instruction"],
                },
                {
                    "prompt": "Condense this wordy reminder: 'all employees must complete timesheets before Friday close of business'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["timesheets"]},
                    "criteria": ["Brief wording"],
                },
                {
                    "prompt": "Make this text concise: 'At this point in time, it is of utmost importance that all employees take into consideration the necessity of completing their timesheets prior to the close of business on Friday.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["timesheets"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Make this text concise in French: 'Submit timesheets before Friday 5 PM'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["vendredi"]},
                    "params": {"language": "fr"},
                    "criteria": ["French concise"],
                },
            ],
        },
        {
            "fam": "fam_rew_polite_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Rewrite this blunt email to be empathetic and polite: 'Your request was rejected because you filled out the wrong form. Do it again.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["apologize", "form", "assist"]},
                    "criteria": ["Polite tone", "Clear explanation", "Courteous"],
                },
                {
                    "prompt": "Make this customer service response courteous and helpful: 'Your request was rejected because you filled out the wrong form.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["form"]},
                    "criteria": ["Empathetic"],
                },
                {
                    "prompt": "Rewrite this blunt email to be empathetic and polite: 'Your request was rejected because you filled out the wrong form. Do it again.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["form"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Rewrite politely in under 25 words",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["form"]},
                    "params": {"max_tokens": 30},
                    "criteria": ["Short length"],
                },
            ],
        },
        {
            "fam": "fam_rew_tech_simplify_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Explain this for non-technical users: 'The distributed transaction failed due to two-phase commit consensus timeout and subsequent cascading rollback.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["computer", "agreement", "undo"]},
                    "params": {"audience": "general"},
                    "criteria": ["Plain English", "No jargon"],
                },
                {
                    "prompt": "Simplify for a beginner: 'distributed transaction failed due to consensus timeout and rollback'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["cancel", "system"]},
                    "params": {"audience": "general"},
                    "criteria": ["Simplified explanation"],
                },
                {
                    "prompt": "Explain this for non-technical users: 'The distributed transaction failed due to two-phase commit consensus timeout and subsequent cascading rollback.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["computer"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Explain this in a 3-bullet analogy for non-technical users",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"-\s*.*-\s*.*-\s*"},
                    "params": {"expected_format": "bullets"},
                    "criteria": ["Bullet analogy"],
                },
            ],
        },
        {
            "fam": "fam_rew_markdown_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Structure these project notes into Markdown with headings # Overview and # Next Steps: 'We finished frontend mockup. Backend API is halfway done. Next we must integrate authentication and deploy to staging.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"#\s*Overview.*#\s*Next Steps"},
                    "params": {"expected_format": "markdown"},
                    "criteria": ["Required headings # Overview and # Next Steps"],
                },
                {
                    "prompt": "Format project notes into Markdown with sections # Overview and # Next Steps",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"#\s*Overview"},
                    "params": {"expected_format": "markdown"},
                    "criteria": ["Markdown structure"],
                },
                {
                    "prompt": "Structure these project notes into Markdown with headings # Overview and # Next Steps: 'We finished frontend mockup. Backend API is halfway done. Next we must integrate authentication and deploy to staging.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"#\s*Overview"},
                    "params": {"expected_format": "markdown"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Format project notes into valid JSON with keys overview and next_steps",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"overview\".*\"next_steps\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON structure"],
                },
            ],
        },
        {
            "fam": "fam_rew_time_news_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Rewrite the breaking news wire from today about the space launch into a social media post",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["launch", "space"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Fresh news rephrase", "Bypasses cache"],
                },
                {
                    "prompt": "Convert today's live space launch update into a 280-character post",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["space"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Short post"],
                },
                {
                    "prompt": "Rewrite the breaking news wire from today about the space launch into a social media post",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["space"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Repeat time sensitive"],
                },
                {
                    "prompt": "Draft a 1-sentence headline for today's breaking space mission",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["mission"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Headline"],
                },
            ],
        },
    ]
    add_family_group("rewriting", dev_rew, test_rew)

    # 4. Short Explanations (40 items, 10 families)
    dev_exp = [
        {
            "fam": "fam_exp_http_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "What does HTTP status code 404 signify in web browsing?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Nn]ot [Ff]ound"},
                    "criteria": ["Identifies Not Found / missing resource"],
                },
                {
                    "prompt": "Explain what a 404 error means when visiting a webpage",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Nn]ot [Ff]ound"},
                    "criteria": ["Not Found meaning"],
                },
                {
                    "prompt": "What does HTTP status code 404 signify in web browsing?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Nn]ot [Ff]ound"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "What does HTTP status code 404 mean in JSON format?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Nn]ot [Ff]ound"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format"],
                },
            ],
        },
        {
            "fam": "fam_exp_photosyn_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Explain photosynthesis in two simple sentences.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["sunlight", "carbon dioxide", "oxygen"]},
                    "criteria": ["Covers sunlight, carbon dioxide, glucose/oxygen"],
                },
                {
                    "prompt": "How do plants convert sunlight into energy? Explain in 2 sentences.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["sunlight", "energy"]},
                    "criteria": ["Photosynthesis mechanism"],
                },
                {
                    "prompt": "Explain photosynthesis in two simple sentences.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["sunlight"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Explain photosynthesis for an elementary school student",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["plant", "sun"]},
                    "params": {"audience": "children"},
                    "criteria": ["Child-friendly tone"],
                },
            ],
        },
        {
            "fam": "fam_exp_dns_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "What is the primary role of a Domain Name System (DNS) server on the internet?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["IP address", "domain name", "translate"]},
                    "criteria": ["Translates human domain names to IP addresses"],
                },
                {
                    "prompt": "How does DNS resolve a website domain to an IP address?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["IP", "domain"]},
                    "criteria": ["DNS resolution"],
                },
                {
                    "prompt": "What is the primary role of a Domain Name System (DNS) server on the internet?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["IP"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Explain DNS in 3 bullet points",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"-\s*.*-\s*.*-\s*"},
                    "params": {"expected_format": "bullets"},
                    "criteria": ["3 bullets"],
                },
            ],
        },
        {
            "fam": "fam_exp_nitrogen_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "What is the boiling point of nitrogen at standard atmospheric pressure?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"-195\.79|77(\.36)?\s*K"},
                    "criteria": ["-195.79°C or 77.36 K"],
                },
                {
                    "prompt": "At what temperature does liquid nitrogen boil?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"-195\.79|77(\.36)?\s*K"},
                    "criteria": ["Accurate boiling point"],
                },
                {
                    "prompt": "What is the boiling point of nitrogen at standard atmospheric pressure?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"-195\.79|77(\.36)?\s*K"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "What is the boiling point of nitrogen in Kelvin?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"77(\.36)?\s*K"},
                    "criteria": ["Kelvin unit 77.36 K"],
                },
            ],
        },
        {
            "fam": "fam_exp_confidential_ip_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Explain the internal routing architecture of the Project Titan quantum simulation pipeline",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Titan", "routing", "pipeline"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Internal project details", "Strict isolation"],
                },
                {
                    "prompt": "Summarize the proprietary Project Titan pipeline design",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Titan"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Proprietary"],
                },
                {
                    "prompt": "Explain the internal routing architecture of the Project Titan quantum simulation pipeline",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Titan"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Repeat private"],
                },
                {
                    "prompt": "Explain Project Titan routing in JSON schema",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Titan"},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "params": {"expected_format": "json"},
                    "criteria": ["Private JSON"],
                },
            ],
        },
    ]

    test_exp = [
        {
            "fam": "fam_exp_ram_ssd_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "What is the fundamental difference between RAM and an SSD in computer hardware?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["volatile", "storage", "speed"]},
                    "criteria": [
                        "RAM is volatile/fast memory; SSD is non-volatile persistent storage"
                    ],
                },
                {
                    "prompt": "Compare RAM and SSD memory: how do they differ?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["volatile", "storage"]},
                    "criteria": ["Volatile vs non-volatile"],
                },
                {
                    "prompt": "What is the fundamental difference between RAM and an SSD in computer hardware?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["volatile"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Explain RAM vs SSD in a 2-row Markdown comparison table",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\|.*RAM.*\|.*SSD.*\|"},
                    "params": {"expected_format": "markdown_table"},
                    "criteria": ["Markdown table"],
                },
            ],
        },
        {
            "fam": "fam_exp_cache_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Why do computer systems and web services utilize caching mechanisms?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["latency", "speed", "database"]},
                    "criteria": ["Reduces latency", "Decreases backend load"],
                },
                {
                    "prompt": "What benefits does caching provide to server applications?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["latency", "load"]},
                    "criteria": ["Benefits of cache"],
                },
                {
                    "prompt": "Why do computer systems and web services utilize caching mechanisms?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["latency"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Explain caching benefits in 3 bullet points",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"-\s*.*-\s*.*-\s*"},
                    "params": {"expected_format": "bullets"},
                    "criteria": ["3 bullets"],
                },
            ],
        },
        {
            "fam": "fam_exp_france_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "What is the capital city of France?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Paris"},
                    "criteria": ["Identifies Paris"],
                },
                {
                    "prompt": "Name the capital of France.",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Paris"},
                    "criteria": ["Paris"],
                },
                {
                    "prompt": "What is the capital city of France?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Paris"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "What is the capital of France and its primary river?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Paris.*Seine"},
                    "criteria": ["Paris and Seine"],
                },
            ],
        },
        {
            "fam": "fam_exp_docker_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Explain how a Docker container differs from a traditional Virtual Machine (VM).",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["kernel", "hypervisor", "guest OS", "isolation"]},
                    "criteria": ["Shared OS kernel vs separate guest OS with hypervisor"],
                },
                {
                    "prompt": "What is the architectural difference between containers and VMs?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["kernel", "hypervisor"]},
                    "criteria": ["Architectural comparison"],
                },
                {
                    "prompt": "Explain how a Docker container differs from a traditional Virtual Machine (VM).",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["kernel"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Explain containers vs VMs in JSON format with pros and cons",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"containers\".*\"vms\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format"],
                },
            ],
        },
        {
            "fam": "fam_exp_time_weather_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "What is the current live outdoor temperature in Central Park New York right now today?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["temperature", "weather", "Central Park"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Live temperature", "Bypasses cache"],
                },
                {
                    "prompt": "Current weather report in Central Park NY right now",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["weather"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Live weather"],
                },
                {
                    "prompt": "What is the current live outdoor temperature in Central Park New York right now today?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["temperature"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Repeat live"],
                },
                {
                    "prompt": "Today's current humidity in Central Park right now",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["humidity"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Humidity"],
                },
            ],
        },
    ]
    add_family_group("short_explanations", dev_exp, test_exp)

    # 5. Summarization (40 items, 10 families)
    dev_sum = [
        {
            "fam": "fam_sum_earnings_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Summarize this quarterly earnings snippet into 2 sentences: 'Initech Corporation recorded Q3 revenue of $14.2M, representing an 8% increase year-over-year. Operating margin expanded to 21% driven by cloud services growth, while customer churn decreased to 1.2%.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["14.2M", "revenue", "cloud"]},
                    "criteria": ["Key metrics included (14.2M, margin, churn)", "Two sentences"],
                },
                {
                    "prompt": "Summarize in 2 sentences: Initech reported $14.2M Q3 revenue, 21% operating margin, 1.2% churn",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["14.2M", "revenue"]},
                    "criteria": ["Concise summary"],
                },
                {
                    "prompt": "Summarize this quarterly earnings snippet into 2 sentences: 'Initech Corporation recorded Q3 revenue of $14.2M, representing an 8% increase year-over-year. Operating margin expanded to 21% driven by cloud services growth, while customer churn decreased to 1.2%.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["14.2M"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Summarize Initech earnings snippet as valid JSON with keys revenue, margin, churn",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"revenue\".*\"margin\".*\"churn\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format"],
                },
            ],
        },
        {
            "fam": "fam_sum_outage_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Summarize this post-mortem into a 1-sentence root cause statement: 'The European cluster experienced an outage lasting 42 minutes when a misconfigured BGP route announcement caused traffic blackholing across three regional ingress gateways.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"BGP.*(?:blackhol|traffic|gateway)"},
                    "criteria": ["Identifies BGP misconfiguration and 42 minute duration"],
                },
                {
                    "prompt": "State the root cause of the European cluster outage in one concise sentence",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"BGP"},
                    "criteria": ["Root cause"],
                },
                {
                    "prompt": "Summarize this post-mortem into a 1-sentence root cause statement: 'The European cluster experienced an outage lasting 42 minutes when a misconfigured BGP route announcement caused traffic blackholing across three regional ingress gateways.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"BGP"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Summarize outage root cause in bullet format with Impact and Cause",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"-\s*Impact.*-\s*Cause"},
                    "params": {"expected_format": "bullets"},
                    "criteria": ["Bullets"],
                },
            ],
        },
        {
            "fam": "fam_sum_climate_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Summarize the main takeaway of this environmental policy: 'The municipal council voted to mandate that all new commercial buildings over 50,000 square feet achieve net-zero operational carbon emissions by 2030, supported by municipal rooftop solar subsidies.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["commercial", "net-zero", "2030"]},
                    "criteria": ["Mandate for 50k+ sq ft buildings to achieve net-zero by 2030"],
                },
                {
                    "prompt": "What is the key rule passed in the municipal environmental policy for commercial buildings?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["net-zero", "2030"]},
                    "criteria": ["Key policy mandate"],
                },
                {
                    "prompt": "Summarize the main takeaway of this environmental policy: 'The municipal council voted to mandate that all new commercial buildings over 50,000 square feet achieve net-zero operational carbon emissions by 2030, supported by municipal rooftop solar subsidies.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["net-zero"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Summarize policy in Spanish in 1 sentence",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["edificios"]},
                    "params": {"language": "es"},
                    "criteria": ["Spanish summary"],
                },
            ],
        },
        {
            "fam": "fam_sum_tos_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Summarize the user data rights in this clause: 'Users retain ownership of submitted content. By submitting content, users grant the platform a non-exclusive, worldwide, royalty-free license to host and display content solely for service operation. Users may delete their data at any time with 30-day purge verification.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["ownership", "delete", "license"]},
                    "criteria": [
                        "Covers user ownership, operational license, 30-day deletion right"
                    ],
                },
                {
                    "prompt": "What rights do users have regarding content and deletion in this Terms clause?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["delete", "ownership"]},
                    "criteria": ["User rights summary"],
                },
                {
                    "prompt": "Summarize the user data rights in this clause: 'Users retain ownership of submitted content. By submitting content, users grant the platform a non-exclusive, worldwide, royalty-free license to host and display content solely for service operation. Users may delete their data at any time with 30-day purge verification.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["ownership"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Summarize user rights as 3 bullet points",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"-\s*.*-\s*.*-\s*"},
                    "params": {"expected_format": "bullets"},
                    "criteria": ["3 bullets"],
                },
            ],
        },
        {
            "fam": "fam_sum_merger_private_01",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Summarize confidential board minutes on proposed acquisition of QuantumAI: 'Board authorized CEO to submit non-binding offer of $450M all-cash, subject to clean IP audit by November 20.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["450M", "QuantumAI", "offer"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Identifies $450M offer for QuantumAI", "Confidential protection"],
                },
                {
                    "prompt": "Provide confidential summary of board decision regarding QuantumAI acquisition",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["450M", "QuantumAI"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Private summary"],
                },
                {
                    "prompt": "Summarize confidential board minutes on proposed acquisition of QuantumAI: 'Board authorized CEO to submit non-binding offer of $450M all-cash, subject to clean IP audit by November 20.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["QuantumAI"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Repeat private"],
                },
                {
                    "prompt": "Summarize QuantumAI board decision in JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"QuantumAI"},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "params": {"expected_format": "json"},
                    "criteria": ["Private JSON"],
                },
            ],
        },
    ]

    test_sum = [
        {
            "fam": "fam_sum_research_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Summarize the key finding of this paper excerpt: 'Evaluating multi-tier routing across 10,000 real queries demonstrated a 68% cost reduction while meeting the 0.70 quality threshold on 94.2% of requests.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["68%", "cost", "94.2%"]},
                    "criteria": ["Mentions 68% cost drop and 94.2% quality satisfaction"],
                },
                {
                    "prompt": "What is the primary conclusion of the multi-tier routing paper excerpt?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["68%", "cost"]},
                    "criteria": ["Key finding"],
                },
                {
                    "prompt": "Summarize the key finding of this paper excerpt: 'Evaluating multi-tier routing across 10,000 real queries demonstrated a 68% cost reduction while meeting the 0.70 quality threshold on 94.2% of requests.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["68%"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Summarize the routing finding in 1 concise sentence",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["cost"]},
                    "criteria": ["Single sentence"],
                },
            ],
        },
        {
            "fam": "fam_sum_medical_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Summarize clinical trial result: 'The Phase III randomized trial of Drug X in 1,200 patients demonstrated a 34% reduction in primary cardiac events (p < 0.001) with no increase in severe adverse reactions compared to placebo.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["34%", "cardiac", "Drug X"]},
                    "criteria": ["Drug X, 34% event reduction, p < 0.001, safety profile"],
                },
                {
                    "prompt": "Summarize Phase III trial outcomes for Drug X",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Drug X", "34%"]},
                    "criteria": ["Trial summary"],
                },
                {
                    "prompt": "Summarize clinical trial result: 'The Phase III randomized trial of Drug X in 1,200 patients demonstrated a 34% reduction in primary cardiac events (p < 0.001) with no increase in severe adverse reactions compared to placebo.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Drug X"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Summarize trial in 2 bullets: Efficacy and Safety",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"-\s*Efficacy.*-\s*Safety"},
                    "params": {"expected_format": "bullets"},
                    "criteria": ["Two bullets"],
                },
            ],
        },
        {
            "fam": "fam_sum_product_launch_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Summarize the launch announcement: 'WidgetPro 2.0 launches next Tuesday featuring real-time collaborative editing, 4K canvas rendering, and offline synchronization for $19/month.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["WidgetPro", "Tuesday", "collaborative"]},
                    "criteria": ["Product name WidgetPro, features, price $19/mo"],
                },
                {
                    "prompt": "What are the core features and price of WidgetPro 2.0?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["WidgetPro", "$19"]},
                    "criteria": ["Features and price"],
                },
                {
                    "prompt": "Summarize the launch announcement: 'WidgetPro 2.0 launches next Tuesday featuring real-time collaborative editing, 4K canvas rendering, and offline synchronization for $19/month.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["WidgetPro"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Summarize WidgetPro launch announcement in JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"WidgetPro"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format"],
                },
            ],
        },
        {
            "fam": "fam_sum_security_patch_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Summarize security advisory: 'CVE-2026-1049 is a remote code execution vulnerability in the OpenSSH parser affecting versions 8.0 through 9.3. Administrators must immediately apply patch 9.4-p1 or disable GSSAPI authentication.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["CVE-2026-1049", "patch", "OpenSSH"]},
                    "criteria": [
                        "CVE ID, vulnerability type RCE, affected versions, remediation patch"
                    ],
                },
                {
                    "prompt": "Provide summary of CVE-2026-1049 advisory and recommended fix",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["CVE-2026-1049", "patch"]},
                    "criteria": ["Fix and advisory"],
                },
                {
                    "prompt": "Summarize security advisory: 'CVE-2026-1049 is a remote code execution vulnerability in the OpenSSH parser affecting versions 8.0 through 9.3. Administrators must immediately apply patch 9.4-p1 or disable GSSAPI authentication.'",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["CVE-2026-1049"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Summarize CVE-2026-1049 in Markdown with # Severity and # Remediation",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"#\s*Severity.*#\s*Remediation"},
                    "params": {"expected_format": "markdown"},
                    "criteria": ["Markdown headings"],
                },
            ],
        },
        {
            "fam": "fam_sum_time_elections_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Summarize today's live municipal election results and vote counting status right now",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["election", "vote", "results"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Live election summary", "Bypasses cache"],
                },
                {
                    "prompt": "What are the live election vote totals reported today right now?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["election"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Live vote totals"],
                },
                {
                    "prompt": "Summarize today's live municipal election results and vote counting status right now",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["election"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Repeat live election"],
                },
                {
                    "prompt": "Draft headline on today's live election results",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["election"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Headline"],
                },
            ],
        },
    ]
    add_family_group("summarization", dev_sum, test_sum)

    # 6. Classification (40 items, 10 families)
    dev_cls = [
        {
            "fam": "fam_cls_sentiment_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Classify the sentiment of this review as POSITIVE, NEGATIVE, or NEUTRAL: 'The screen resolution is breathtaking and battery life lasts all day!'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bPOSITIVE\b"},
                    "criteria": ["Identifies POSITIVE"],
                },
                {
                    "prompt": "Determine sentiment (POSITIVE, NEGATIVE, NEUTRAL): 'The screen resolution is breathtaking and battery life lasts all day!'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bPOSITIVE\b"},
                    "criteria": ["POSITIVE"],
                },
                {
                    "prompt": "Classify the sentiment of this review as POSITIVE, NEGATIVE, or NEUTRAL: 'The screen resolution is breathtaking and battery life lasts all day!'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bPOSITIVE\b"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Classify review sentiment in JSON format: {'sentiment': '...'}",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"sentiment\"\s*:\s*\"POSITIVE\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format POSITIVE"],
                },
            ],
        },
        {
            "fam": "fam_cls_intent_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Classify the customer support intent as BILLING, TECHNICAL, or ACCOUNT: 'I was double charged on my invoice this morning.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bBILLING\b"},
                    "criteria": ["Identifies BILLING intent"],
                },
                {
                    "prompt": "What is the intent category (BILLING, TECHNICAL, ACCOUNT): 'I was double charged on my invoice'?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bBILLING\b"},
                    "criteria": ["BILLING"],
                },
                {
                    "prompt": "Classify the customer support intent as BILLING, TECHNICAL, or ACCOUNT: 'I was double charged on my invoice this morning.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bBILLING\b"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Classify support ticket intent in JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"intent\"\s*:\s*\"BILLING\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON BILLING"],
                },
            ],
        },
        {
            "fam": "fam_cls_urgency_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Classify ticket urgency as P1_CRITICAL, P2_HIGH, or P3_NORMAL: 'All production servers are returning 500 errors and our checkout is completely offline.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"P1_CRITICAL"},
                    "criteria": ["Identifies P1_CRITICAL"],
                },
                {
                    "prompt": "Assign priority (P1_CRITICAL, P2_HIGH, P3_NORMAL) to total production checkout outage",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"P1_CRITICAL"},
                    "criteria": ["P1_CRITICAL"],
                },
                {
                    "prompt": "Classify ticket urgency as P1_CRITICAL, P2_HIGH, or P3_NORMAL: 'All production servers are returning 500 errors and our checkout is completely offline.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"P1_CRITICAL"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Classify urgency in JSON with rationale",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"P1_CRITICAL"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format"],
                },
            ],
        },
        {
            "fam": "fam_cls_spam_01",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Classify email as SPAM or HAM: 'Congratulations! You have won $1,000,000 in the overseas lottery! Click this link to claim now!'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bSPAM\b"},
                    "criteria": ["Identifies SPAM"],
                },
                {
                    "prompt": "Is this email SPAM or HAM: 'Lottery winner claim your prize immediately'?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bSPAM\b"},
                    "criteria": ["SPAM"],
                },
                {
                    "prompt": "Classify email as SPAM or HAM: 'Congratulations! You have won $1,000,000 in the overseas lottery! Click this link to claim now!'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bSPAM\b"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Classify as SPAM or HAM with confidence score between 0 and 1 in JSON",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"label\"\s*:\s*\"SPAM\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON SPAM"],
                },
            ],
        },
        {
            "fam": "fam_cls_privacy_priv_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Classify internal patient healthcare record sensitivity as PUBLIC, INTERNAL, or RESTRICTED_PHI",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"RESTRICTED_PHI"},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Identifies RESTRICTED_PHI"],
                },
                {
                    "prompt": "Determine compliance privacy classification of hospital patient diagnostic record",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"RESTRICTED_PHI"},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["RESTRICTED_PHI"],
                },
                {
                    "prompt": "Classify internal patient healthcare record sensitivity as PUBLIC, INTERNAL, or RESTRICTED_PHI",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"RESTRICTED_PHI"},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Repeat private"],
                },
                {
                    "prompt": "Classify hospital record privacy in JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"RESTRICTED_PHI"},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "params": {"expected_format": "json"},
                    "criteria": ["Private JSON"],
                },
            ],
        },
    ]

    test_cls = [
        {
            "fam": "fam_cls_topic_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Classify article topic as HEALTHCARE, FINANCE, or ASTRONOMY: 'James Webb Space Telescope captures ancient galaxy clusters 13 billion light years away.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bASTRONOMY\b"},
                    "criteria": ["Identifies ASTRONOMY"],
                },
                {
                    "prompt": "What is the topic domain (HEALTHCARE, FINANCE, ASTRONOMY) of telescope discovery?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bASTRONOMY\b"},
                    "criteria": ["ASTRONOMY"],
                },
                {
                    "prompt": "Classify article topic as HEALTHCARE, FINANCE, or ASTRONOMY: 'James Webb Space Telescope captures ancient galaxy clusters 13 billion light years away.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bASTRONOMY\b"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Classify topic in JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"topic\"\s*:\s*\"ASTRONOMY\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON ASTRONOMY"],
                },
            ],
        },
        {
            "fam": "fam_cls_lang_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Identify the natural language of this sentence: 'Guten Tag, wie geht es Ihnen heute?'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Gg]erman|[Dd]eutsch"},
                    "criteria": ["Identifies German"],
                },
                {
                    "prompt": "What language is this: 'Guten Tag, wie geht es Ihnen heute?'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Gg]erman|[Dd]eutsch"},
                    "criteria": ["German"],
                },
                {
                    "prompt": "Identify the natural language of this sentence: 'Guten Tag, wie geht es Ihnen heute?'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Gg]erman|[Dd]eutsch"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Identify language and return ISO-639-1 code in JSON",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\"de\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON code 'de'"],
                },
            ],
        },
        {
            "fam": "fam_cls_feedback_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Classify feedback as PRAISE, BUG, or FEATURE_REQUEST: 'Could you please add dark mode support to the desktop application?'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"FEATURE_REQUEST"},
                    "criteria": ["Identifies FEATURE_REQUEST"],
                },
                {
                    "prompt": "What category is user asking for dark mode: PRAISE, BUG, or FEATURE_REQUEST?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"FEATURE_REQUEST"},
                    "criteria": ["FEATURE_REQUEST"],
                },
                {
                    "prompt": "Classify feedback as PRAISE, BUG, or FEATURE_REQUEST: 'Could you please add dark mode support to the desktop application?'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"FEATURE_REQUEST"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Classify feedback in JSON format with priority",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"FEATURE_REQUEST"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format"],
                },
            ],
        },
        {
            "fam": "fam_cls_neg_sentiment_02",
            "tier": "easy",
            "items": [
                {
                    "prompt": "Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL: 'The device overheated within ten minutes and completely stopped turning on.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bNEGATIVE\b"},
                    "criteria": ["Identifies NEGATIVE"],
                },
                {
                    "prompt": "Determine sentiment: 'device overheated within ten minutes and stopped working'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bNEGATIVE\b"},
                    "criteria": ["NEGATIVE"],
                },
                {
                    "prompt": "Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL: 'The device overheated within ten minutes and completely stopped turning on.'",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\bNEGATIVE\b"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Classify sentiment in JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"sentiment\"\s*:\s*\"NEGATIVE\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON NEGATIVE"],
                },
            ],
        },
        {
            "fam": "fam_cls_time_breaking_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Classify the severity of today's live volcanic eruption advisory in Iceland as ADVISORY, WATCH, or WARNING",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["volcanic", "Iceland"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Live volcanic classification", "Bypasses cache"],
                },
                {
                    "prompt": "What is the official weather alert severity for today's active Iceland eruption?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Iceland"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Live alert"],
                },
                {
                    "prompt": "Classify the severity of today's live volcanic eruption advisory in Iceland as ADVISORY, WATCH, or WARNING",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Iceland"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Repeat live alert"],
                },
                {
                    "prompt": "Classify Iceland eruption alert in JSON format",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Iceland"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON alert"],
                },
            ],
        },
    ]
    add_family_group("classification", dev_cls, test_cls)

    # 7. Multi-step Reasoning (40 items, 10 families)
    dev_rsn = [
        {
            "fam": "fam_rsn_itinerary_01",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Flight A lands in Chicago at 14:15 CDT. Flight B departs at 16:45 CDT. The passenger needs 45 minutes for customs and 30 minutes to transfer terminals. What is the remaining layover buffer time in minutes?",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 75.0, "tolerance": 0.0},
                    "criteria": ["Layover = 150 mins; buffer = 150 - 75 = 75 minutes"],
                },
                {
                    "prompt": "Calculate remaining layover buffer: land at 14:15, depart at 16:45, customs takes 45m, transfer takes 30m.",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 75.0, "tolerance": 0.0},
                    "criteria": ["75 minutes buffer"],
                },
                {
                    "prompt": "Flight A lands in Chicago at 14:15 CDT. Flight B departs at 16:45 CDT. The passenger needs 45 minutes for customs and 30 minutes to transfer terminals. What is the remaining layover buffer time in minutes?",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 75.0, "tolerance": 0.0},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Calculate layover buffer and explain steps in JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"75"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format with 75"],
                },
            ],
        },
        {
            "fam": "fam_rsn_apples_01",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Three boxes are labeled APPLES, ORANGES, and APPLES_AND_ORANGES. Every label is incorrect. You pick one fruit from the box labeled APPLES_AND_ORANGES and it is an Apple. How should all three boxes be correctly labeled?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Apples", "Oranges", "Apples and Oranges"]},
                    "criteria": [
                        "Deduction: Picked box is Apples; box labeled Oranges is Apples and Oranges; box labeled Apples is Oranges"
                    ],
                },
                {
                    "prompt": "Solve the three mislabeled fruit boxes puzzle where drawing an apple from the mixed label identifies the boxes",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Apples", "Oranges"]},
                    "criteria": ["Correct logic solution"],
                },
                {
                    "prompt": "Three boxes are labeled APPLES, ORANGES, and APPLES_AND_ORANGES. Every label is incorrect. You pick one fruit from the box labeled APPLES_AND_ORANGES and it is an Apple. How should all three boxes be correctly labeled?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Apples"]},
                    "criteria": ["Repeat puzzle"],
                },
                {
                    "prompt": "Solve the mislabeled fruit puzzle in 3 numbered deduction steps",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"1\..*2\..*3\."},
                    "params": {"expected_format": "numbered"},
                    "criteria": ["Numbered steps"],
                },
            ],
        },
        {
            "fam": "fam_rsn_budget_01",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Allocate a $100,000 budget across 3 projects: A, B, C. Project A must receive twice as much as B. Project C must receive $10,000 more than B. What is the exact dollar allocation for each project?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"A:?\s*\$?45,?000.*B:?\s*\$?22,?500.*C:?\s*\$?32,?500"},
                    "criteria": ["A = $45,000, B = $22,500, C = $32,500"],
                },
                {
                    "prompt": "Solve algebraic budget allocation: 2x + x + (x + 10,000) = 100,000 for projects A, B, C",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"45,?000.*22,?500.*32,?500"},
                    "criteria": ["Accurate allocation values"],
                },
                {
                    "prompt": "Allocate a $100,000 budget across 3 projects: A, B, C. Project A must receive twice as much as B. Project C must receive $10,000 more than B. What is the exact dollar allocation for each project?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"45,?000"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Provide the budget allocation in valid JSON format: {'A': 45000, 'B': 22500, 'C': 32500}",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"A\"\s*:\s*45000"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON allocation"],
                },
            ],
        },
        {
            "fam": "fam_rsn_code_race_01",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Analyze this pseudocode for concurrency defects: 'thread1: read(x); x = x + 1; write(x);' and 'thread2: read(x); x = x + 2; write(x);'. If initial x = 0, identify all possible final values of x.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["1", "2", "3", "race condition"]},
                    "criteria": ["Identifies possible values 1, 2, and 3 due to lost updates"],
                },
                {
                    "prompt": "Find all potential final values of x in concurrent increments +1 and +2 without mutex synchronization",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["1", "2", "3"]},
                    "criteria": ["Values 1, 2, 3"],
                },
                {
                    "prompt": "Analyze this pseudocode for concurrency defects: 'thread1: read(x); x = x + 1; write(x);' and 'thread2: read(x); x = x + 2; write(x);'. If initial x = 0, identify all possible final values of x.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["race condition"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Explain the race condition in Markdown with # Vulnerability and # Mitigation",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"#\s*Vulnerability.*#\s*Mitigation"},
                    "params": {"expected_format": "markdown"},
                    "criteria": ["Markdown headings"],
                },
            ],
        },
        {
            "fam": "fam_rsn_audit_priv_01",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Perform forensic reconciliation of internal ledger accounts for entity Omega Corp: Total recorded disbursements $1,240,000, bank statement withdrawals $1,285,000. Identify missing variance.",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 45000.0, "tolerance": 0.0},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Variance = $45,000", "Strict confidentiality"],
                },
                {
                    "prompt": "Calculate internal ledger variance: disbursements 1,240,000 vs withdrawals 1,285,000 for Omega Corp",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 45000.0, "tolerance": 0.0},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Variance 45,000"],
                },
                {
                    "prompt": "Perform forensic reconciliation of internal ledger accounts for entity Omega Corp: Total recorded disbursements $1,240,000, bank statement withdrawals $1,285,000. Identify missing variance.",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 45000.0, "tolerance": 0.0},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Repeat private audit"],
                },
                {
                    "prompt": "Provide reconciliation output in JSON format with variance field",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"45,?000"},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "params": {"expected_format": "json"},
                    "criteria": ["Private JSON"],
                },
            ],
        },
    ]

    test_rsn = [
        {
            "fam": "fam_rsn_trains_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Train A departs City X heading to City Y at 60 mph. Train B departs City Y heading to City X at 90 mph along the same 300-mile track. How many hours until they meet?",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 2.0, "tolerance": 0.01},
                    "criteria": ["Combined speed = 150 mph; time = 300 / 150 = 2 hours"],
                },
                {
                    "prompt": "Calculate collision time for two trains 300 miles apart traveling at 60 mph and 90 mph toward each other",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 2.0, "tolerance": 0.01},
                    "criteria": ["2 hours"],
                },
                {
                    "prompt": "Train A departs City X heading to City Y at 60 mph. Train B departs City Y heading to City X at 90 mph along the same 300-mile track. How many hours until they meet?",
                    "checker_type": "numeric_tolerance",
                    "cfg": {"expected": 2.0, "tolerance": 0.01},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Solve train distance problem in JSON format with meeting_time_hours: 2",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\"meeting_time_hours\"\s*:\s*2"},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format"],
                },
            ],
        },
        {
            "fam": "fam_rsn_syllogism_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Premise 1: All mammals are warm-blooded. Premise 2: All whales are mammals. Premise 3: Beluga is a whale. What logically necessary conclusion follows about Beluga?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Bb]eluga is warm-blooded"},
                    "criteria": ["Deduction: Beluga is warm-blooded"],
                },
                {
                    "prompt": "Deduce conclusion from: all mammals warm-blooded, all whales mammals, Beluga is whale",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Bb]eluga is warm-blooded"},
                    "criteria": ["Beluga warm-blooded"],
                },
                {
                    "prompt": "Premise 1: All mammals are warm-blooded. Premise 2: All whales are mammals. Premise 3: Beluga is a whale. What logically necessary conclusion follows about Beluga?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Bb]eluga is warm-blooded"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "State the deductive conclusion in a single formal proposition sentence",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Bb]eluga is warm-blooded"},
                    "criteria": ["Formal proposition"],
                },
            ],
        },
        {
            "fam": "fam_rsn_prob_02",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "You roll two fair six-sided dice. What is the exact theoretical probability of rolling a sum equal to 7 or 11?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"8/36|2/9|0\.222"},
                    "criteria": ["Sum 7 (6 outcomes) + Sum 11 (2 outcomes) = 8/36 = 2/9 (~0.222)"],
                },
                {
                    "prompt": "Calculate probability of rolling sum 7 or 11 with two standard dice",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"2/9|8/36|0\.222"},
                    "criteria": ["Probability 2/9"],
                },
                {
                    "prompt": "You roll two fair six-sided dice. What is the exact theoretical probability of rolling a sum equal to 7 or 11?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"2/9|8/36|0\.222"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Calculate dice sum 7 or 11 probability and express as percentage",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"22\.2%?"},
                    "criteria": ["Percentage ~22.2%"],
                },
            ],
        },
        {
            "fam": "fam_rsn_scheduling_02",
            "tier": "moderate",
            "items": [
                {
                    "prompt": "Person A is free from 9:00 to 11:30. Person B is free from 10:00 to 13:00. Person C is free from 10:30 to 12:00. Find the maximum overlapping meeting window common to all three people.",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"10:30\s*(?:to|-)\s*11:30"},
                    "criteria": ["Overlap window: 10:30 to 11:30 (60 minutes)"],
                },
                {
                    "prompt": "Find common free time for Person A (9-11:30), B (10-13), C (10:30-12)",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"10:30\s*(?:to|-)\s*11:30"},
                    "criteria": ["10:30 to 11:30"],
                },
                {
                    "prompt": "Person A is free from 9:00 to 11:30. Person B is free from 10:00 to 13:00. Person C is free from 10:30 to 12:00. Find the maximum overlapping meeting window common to all three people.",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"10:30"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Output common meeting slot in JSON format: {'start': '10:30', 'end': '11:30'}",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\"start\"\s*:\s*\"10:30\".*\"end\"\s*:\s*\"11:30\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON slot"],
                },
            ],
        },
        {
            "fam": "fam_rsn_bottleneck_02",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "A manufacturing pipeline has 4 sequential stations: Stamping (120 units/hr), Welding (85 units/hr), Painting (95 units/hr), Assembly (110 units/hr). Identify the bottleneck station and maximum pipeline throughput.",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Ww]elding.*85"},
                    "criteria": ["Bottleneck is Welding with capacity 85 units/hour"],
                },
                {
                    "prompt": "Which station limits throughput: Stamping (120/h), Welding (85/h), Painting (95/h), Assembly (110/h)?",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Ww]elding"},
                    "criteria": ["Welding"],
                },
                {
                    "prompt": "A manufacturing pipeline has 4 sequential stations: Stamping (120 units/hr), Welding (85 units/hr), Painting (95 units/hr), Assembly (110 units/hr). Identify the bottleneck station and maximum pipeline throughput.",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"[Ww]elding"},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Identify bottleneck station in JSON format with station and max_units_per_hour",
                    "checker_type": "regex",
                    "cfg": {
                        "pattern": r"\{\s*\"station\"\s*:\s*\"[Ww]elding\".*\"max_units_per_hour\"\s*:\s*85"
                    },
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON bottleneck"],
                },
            ],
        },
    ]
    add_family_group("multi_step_reasoning", dev_rsn, test_rsn)

    # 8. Synthesis (40 items, 10 families)
    dev_syn = [
        {
            "fam": "fam_syn_db_comp_01",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Synthesize a structured technical comparison between PostgreSQL and MongoDB evaluating data modeling, ACID transaction guarantees, and horizontal scaling capabilities.",
                    "checker_type": "rubric",
                    "cfg": {
                        "required_keywords": [
                            "ACID",
                            "relational",
                            "document",
                            "sharding",
                            "PostgreSQL",
                            "MongoDB",
                        ]
                    },
                    "criteria": [
                        "Compares relational schema vs document model",
                        "ACID multi-document guarantees",
                        "Horizontal scaling tradeoffs",
                    ],
                },
                {
                    "prompt": "Compare PostgreSQL vs MongoDB across data model, ACID transactions, and sharding scalability",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["PostgreSQL", "MongoDB", "ACID"]},
                    "criteria": ["Technical comparison"],
                },
                {
                    "prompt": "Synthesize a structured technical comparison between PostgreSQL and MongoDB evaluating data modeling, ACID transaction guarantees, and horizontal scaling capabilities.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["PostgreSQL", "MongoDB"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Compare PostgreSQL and MongoDB in Markdown with headings # Data Model, # Transactions, and # Scalability",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"#\s*Data Model.*#\s*Transactions.*#\s*Scalability"},
                    "params": {"expected_format": "markdown"},
                    "criteria": ["Required Markdown headings"],
                },
            ],
        },
        {
            "fam": "fam_syn_policy_01",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Synthesize three user research findings (high friction on onboarding, confusing pricing tiers, fast customer support) into two actionable product roadmap recommendations.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["onboarding", "pricing", "recommendation"]},
                    "criteria": [
                        "Simplifying onboarding",
                        "Streamlining pricing tiers",
                        "Preserving support quality",
                    ],
                },
                {
                    "prompt": "Synthesize product roadmap actions from onboarding friction, pricing confusion, and good support findings",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["onboarding", "pricing"]},
                    "criteria": ["Actionable recommendations"],
                },
                {
                    "prompt": "Synthesize three user research findings (high friction on onboarding, confusing pricing tiers, fast customer support) into two actionable product roadmap recommendations.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["onboarding"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Provide the two product recommendations in JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"recommendations\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format"],
                },
            ],
        },
        {
            "fam": "fam_syn_ha_arch_01",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Design a high-availability cloud architecture proposal for an e-commerce API requiring multi-region failover, 99.99% uptime, and sub-100ms read latency.",
                    "checker_type": "rubric",
                    "cfg": {
                        "required_keywords": [
                            "multi-region",
                            "load balancer",
                            "replication",
                            "cache",
                            "database",
                        ]
                    },
                    "criteria": [
                        "Multi-region routing",
                        "Read replica caching",
                        "Active-passive or active-active failover",
                    ],
                },
                {
                    "prompt": "Propose high-availability cloud architecture for 99.99% SLA e-commerce backend",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["load balancer", "failover"]},
                    "criteria": ["Architecture design"],
                },
                {
                    "prompt": "Design a high-availability cloud architecture proposal for an e-commerce API requiring multi-region failover, 99.99% uptime, and sub-100ms read latency.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["multi-region"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Design cloud architecture proposal in Markdown with # Components and # Failover Strategy",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"#\s*Components.*#\s*Failover Strategy"},
                    "params": {"expected_format": "markdown"},
                    "criteria": ["Markdown headings"],
                },
            ],
        },
        {
            "fam": "fam_syn_remote_work_01",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Synthesize an objective, balanced synthesis of empirical arguments for and against permanent remote work policies in engineering organizations.",
                    "checker_type": "rubric",
                    "cfg": {
                        "required_keywords": [
                            "productivity",
                            "communication",
                            "isolation",
                            "retention",
                        ]
                    },
                    "criteria": [
                        "Benefits (retention, focus)",
                        "Drawbacks (onboarding, serendipity, isolation)",
                    ],
                },
                {
                    "prompt": "Provide balanced analysis of pros and cons of remote work for software engineering teams",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["productivity", "collaboration"]},
                    "criteria": ["Balanced synthesis"],
                },
                {
                    "prompt": "Synthesize an objective, balanced synthesis of empirical arguments for and against permanent remote work policies in engineering organizations.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["productivity"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Synthesize remote work debate into 2 columns: Advantages and Disadvantages",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Advantages.*Disadvantages"},
                    "params": {"expected_format": "markdown"},
                    "criteria": ["Two columns"],
                },
            ],
        },
        {
            "fam": "fam_syn_internal_strat_01",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Synthesize confidential corporate pivot strategy for Project Mercury moving from B2C subscriptions to Enterprise licensing",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Mercury", "Enterprise", "licensing"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Strategic pivot B2C to Enterprise", "Strict isolation"],
                },
                {
                    "prompt": "Draft strategic memo on Project Mercury enterprise pivot",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Mercury", "enterprise"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Confidential memo"],
                },
                {
                    "prompt": "Synthesize confidential corporate pivot strategy for Project Mercury moving from B2C subscriptions to Enterprise licensing",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Mercury"]},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "criteria": ["Repeat private strategy"],
                },
                {
                    "prompt": "Summarize Project Mercury enterprise pivot in JSON format with timeline and revenue_model",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"Mercury"},
                    "privacy": PrivacyClass.CONFIDENTIAL_PRIVATE,
                    "params": {"expected_format": "json"},
                    "criteria": ["Private JSON"],
                },
            ],
        },
    ]

    test_syn = [
        {
            "fam": "fam_syn_biomimetic_02",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Synthesize concepts from avian flocking behavior and swarm robotics to propose a collision-avoidance protocol for autonomous drone fleets.",
                    "checker_type": "rubric",
                    "cfg": {
                        "required_keywords": [
                            "alignment",
                            "separation",
                            "cohesion",
                            "collision",
                            "drone",
                        ]
                    },
                    "criteria": [
                        "Reynolds boids principles applied to drone flocking",
                        "Decentralized consensus",
                    ],
                },
                {
                    "prompt": "How can bird flocking rules be synthesized into drone swarm collision prevention algorithms?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["flocking", "drone", "collision"]},
                    "criteria": ["Biomimetic synthesis"],
                },
                {
                    "prompt": "Synthesize concepts from avian flocking behavior and swarm robotics to propose a collision-avoidance protocol for autonomous drone fleets.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["flocking"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Propose drone collision protocol in Markdown with # Biological Inspiration and # Protocol Rules",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"#\s*Biological Inspiration.*#\s*Protocol Rules"},
                    "params": {"expected_format": "markdown"},
                    "criteria": ["Required headings"],
                },
            ],
        },
        {
            "fam": "fam_syn_energy_grid_02",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Synthesize a proposal integrating intermittent solar energy, grid-scale battery storage, and dynamic time-of-use pricing to stabilize municipal peak power demand.",
                    "checker_type": "rubric",
                    "cfg": {
                        "required_keywords": [
                            "solar",
                            "battery",
                            "peak demand",
                            "pricing",
                            "storage",
                        ]
                    },
                    "criteria": [
                        "Combines renewable generation, storage dispatch, and demand-side economic incentives"
                    ],
                },
                {
                    "prompt": "Design peak demand stabilization strategy combining solar, battery storage, and dynamic pricing",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["solar", "storage", "pricing"]},
                    "criteria": ["Grid stabilization synthesis"],
                },
                {
                    "prompt": "Synthesize a proposal integrating intermittent solar energy, grid-scale battery storage, and dynamic time-of-use pricing to stabilize municipal peak power demand.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["solar"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Summarize grid proposal in 3 actionable policy steps",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"1\..*2\..*3\."},
                    "params": {"expected_format": "numbered"},
                    "criteria": ["Numbered steps"],
                },
            ],
        },
        {
            "fam": "fam_syn_llm_eval_02",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Synthesize the methodological challenges of using LLMs as evaluators for other LLMs, addressing circularity, judge model bias, self-preference, and position bias.",
                    "checker_type": "rubric",
                    "cfg": {
                        "required_keywords": [
                            "self-preference",
                            "position bias",
                            "circularity",
                            "judge",
                        ]
                    },
                    "criteria": [
                        "Analyzes self-enhancement bias, order effects, sensitivity to rubrics"
                    ],
                },
                {
                    "prompt": "What are the primary biases and validity risks when using LLM-as-a-judge for evaluation benchmarks?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["bias", "judge", "position"]},
                    "criteria": ["Judge bias analysis"],
                },
                {
                    "prompt": "Synthesize the methodological challenges of using LLMs as evaluators for other LLMs, addressing circularity, judge model bias, self-preference, and position bias.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["self-preference"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Synthesize LLM-as-a-judge challenges in Markdown with # Biases and # Mitigations",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"#\s*Biases.*#\s*Mitigations"},
                    "params": {"expected_format": "markdown"},
                    "criteria": ["Markdown headings"],
                },
            ],
        },
        {
            "fam": "fam_syn_api_gateway_02",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Synthesize architectural design tradeoffs between centralized API Gateways versus decentralized service-mesh sidecar proxies in Kubernetes microservices.",
                    "checker_type": "rubric",
                    "cfg": {
                        "required_keywords": [
                            "latency",
                            "sidecar",
                            "gateway",
                            "service mesh",
                            "Envoy",
                        ]
                    },
                    "criteria": [
                        "Gateway handles edge routing/auth; sidecars handle east-west traffic and fine-grained mTLS",
                        "Resource overhead tradeoffs",
                    ],
                },
                {
                    "prompt": "Compare API Gateway vs Service Mesh sidecar architectures in Kubernetes",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["gateway", "sidecar", "mesh"]},
                    "criteria": ["Architectural tradeoffs"],
                },
                {
                    "prompt": "Synthesize architectural design tradeoffs between centralized API Gateways versus decentralized service-mesh sidecar proxies in Kubernetes microservices.",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["gateway"]},
                    "criteria": ["Repeat"],
                },
                {
                    "prompt": "Provide gateway vs mesh tradeoffs in JSON format",
                    "checker_type": "regex",
                    "cfg": {"pattern": r"\{\s*\"gateway\".*\"service_mesh\""},
                    "params": {"expected_format": "json"},
                    "criteria": ["JSON format"],
                },
            ],
        },
        {
            "fam": "fam_syn_time_markets_02",
            "tier": "difficult",
            "items": [
                {
                    "prompt": "Synthesize today's live macroeconomic indicators and Federal Reserve rate decisions announced this morning",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Federal Reserve", "interest rate", "inflation"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Live macroeconomic synthesis", "Bypasses cache"],
                },
                {
                    "prompt": "What are today's live interest rate announcements and market impact right now?",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["rate", "market"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Live market synthesis"],
                },
                {
                    "prompt": "Synthesize today's live macroeconomic indicators and Federal Reserve rate decisions announced this morning",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["Federal Reserve"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Repeat live macro"],
                },
                {
                    "prompt": "Draft live market headline for today's rate announcement",
                    "checker_type": "rubric",
                    "cfg": {"required_keywords": ["rate"]},
                    "time_sensitive": True,
                    "cache_eligibility": False,
                    "criteria": ["Headline"],
                },
            ],
        },
    ]
    add_family_group("synthesis", dev_syn, test_syn)

    return {
        "version": "1.0.0",
        "description": "InferPath Pilot Evaluation Benchmark Dataset (320 items across 8 categories and 80 leakage-free families)",
        "created_date": "2026-10-13",
        "items": items,
    }


def main():
    dataset = build_dataset()
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "pilot_dataset.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(dataset['items'])} items to {out_file}.")


if __name__ == "__main__":
    main()

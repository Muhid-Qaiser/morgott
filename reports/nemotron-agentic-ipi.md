# Nemotron agentic indirect-injection audit

Generated: 2026-07-20T06:58:21+00:00

This is evaluation-only system/indirect evidence. The source is fully synthetic, positive-only, and filtered for attacks that succeeded against its defender; it is not benign-utility, false-positive, or production evidence.

## Pinned source and safe projection

- Revision: `d738d4f361cc38bb4d7a42b9066776dade5332f5`
- License: CC-BY-4.0
- File SHA-256: `3329da17564a7eb287e2730fc7d6956e1f4fe51e8950ac4f110b3c37e78cf3b9`
- Raw source rows / exact-unique retained texts: 1272 / 676
- Exact duplicates removed / exact fit overlaps blocked: 596 / 0
- Persisted content: `injection.injection_text` plus domain, attack category, injection vector, and target-tool grouping metadata.
- Omitted: source environment and its synthetic identity records, system/user prompts, tool schemas, injection goal, target arguments, and provider/model responses. The public synthetic injection text itself is retained because it is the detector input.

## Leakage audit

Across every other processed fit/evaluation output, 0 retained source texts have an exact normalized match and 0 have a non-exact near match at TF-IDF cosine >= 0.90. Fit-only counts are 0 exact and 0 near. Exact fit overlaps are blocked by the data builder; fuzzy/evaluation matches remain disclosed and the suite stays separate.

## Detector recall

| Signal | Elevated / rows | Recall |
|---|---:|---:|
| direct_user_fallback_diagnostic | 0 / 676 | 0.0000 |
| indirect_sensor | 0 / 676 | 0.0000 |
| combined_untrusted_shadow | 0 / 676 | 0.0000 |

## Validation-anchored operating-point diagnostics

Every threshold was selected on the original grouped direct or BIPIA validation rows before scoring this positive-only source.

| Direct validation FPR budget | Threshold | Observed validation FPR | Source recall |
|---:|---:|---:|---:|
| 0.1% | 0.871110 | 0.0010 | 0.0000 |
| 0.5% | 0.534376 | 0.0049 | 0.0015 |
| 1.0% | 0.391502 | 0.0100 | 0.0296 |
| 2.0% | 0.278024 | 0.0199 | 0.1109 |
| 5.0% | 0.164020 | 0.0500 | 0.4512 |

| Indirect BIPIA-validation FPR budget | Threshold | Observed validation FPR | Validation recall | Source recall |
|---:|---:|---:|---:|---:|
| 0.0% | 0.778259 | 0.0000 | 0.6667 | 0.0000 |
| 1.0% | 0.746734 | 0.0070 | 0.6667 | 0.0000 |
| 2.0% | 0.737293 | 0.0141 | 0.7024 | 0.0000 |
| 5.0% | 0.679236 | 0.0493 | 0.7143 | 0.0148 |
| 10.0% | 0.603295 | 0.0986 | 0.8571 | 0.5414 |
| 20.0% | 0.260315 | 0.1972 | 1.0000 | 1.0000 |

| Per-component validation FPR budget | Combined source recall |
|---:|---:|
| 1.0% | 0.0296 |
| 2.0% | 0.1109 |
| 5.0% | 0.4601 |

The combined rows do not estimate a combined FPR: the two component budgets come from different validation mixtures and OR aggregation accumulates false signals. The source did not choose any threshold.

### direct_user_fallback_diagnostic by source grouping


#### domain

| Value | Elevated / rows | Recall |
|---|---:|---:|
| ecommerce | 0 / 89 | 0.0000 |
| education | 0 / 93 | 0.0000 |
| healthcare | 0 / 156 | 0.0000 |
| hr | 0 / 104 | 0.0000 |
| insurance | 0 / 28 | 0.0000 |
| it_helpdesk | 0 / 36 | 0.0000 |
| legal | 0 / 14 | 0.0000 |
| logistics | 0 / 72 | 0.0000 |
| real_estate | 0 / 84 | 0.0000 |

#### attack_category

| Value | Elevated / rows | Recall |
|---|---:|---:|
| data_modification | 0 / 203 | 0.0000 |
| denial_of_service | 0 / 137 | 0.0000 |
| exfiltration | 0 / 79 | 0.0000 |
| unauthorized_action | 0 / 257 | 0.0000 |

#### injection_vector

| Value | Elevated / rows | Recall |
|---|---:|---:|
| adjuster_note_text | 0 / 4 | 0.0000 |
| advisor_notes | 0 / 22 | 0.0000 |
| application_notes | 0 / 37 | 0.0000 |
| appointment_reason | 0 / 36 | 0.0000 |
| asset_notes | 0 / 11 | 0.0000 |
| carrier_message_body | 0 / 1 | 0.0000 |
| case_notes | 0 / 6 | 0.0000 |
| case_summary | 0 / 3 | 0.0000 |
| chart_notes | 0 / 59 | 0.0000 |
| claim_description | 0 / 13 | 0.0000 |
| client_notes | 0 / 18 | 0.0000 |
| course_description | 0 / 20 | 0.0000 |
| court_location | 0 / 2 | 0.0000 |
| court_notes | 0 / 3 | 0.0000 |
| customs_additional_info | 0 / 16 | 0.0000 |
| delivery_special_instructions | 0 / 12 | 0.0000 |
| fraud_flag_reason | 0 / 4 | 0.0000 |
| grader_comments | 0 / 2 | 0.0000 |
| incident_description | 0 / 20 | 0.0000 |
| inspection_findings | 0 / 21 | 0.0000 |
| interview_feedback | 0 / 12 | 0.0000 |
| job_description | 0 / 25 | 0.0000 |
| kb_article_content | 0 / 5 | 0.0000 |
| lab_comments | 0 / 44 | 0.0000 |
| message_body | 0 / 78 | 0.0000 |
| order_notes | 0 / 34 | 0.0000 |
| payout_notes | 0 / 1 | 0.0000 |
| policy_notes | 0 / 6 | 0.0000 |
| product_description | 0 / 30 | 0.0000 |
| property_description | 0 / 23 | 0.0000 |
| resolution_notes | 0 / 11 | 0.0000 |
| resume_text | 0 / 4 | 0.0000 |
| reviewer_notes | 0 / 36 | 0.0000 |
| shipment_notes | 0 / 22 | 0.0000 |
| ticket_description | 0 / 14 | 0.0000 |
| warehouse_comments | 0 / 21 | 0.0000 |

#### target_tool

| Value | Elevated / rows | Recall |
|---|---:|---:|
| add_adjuster_note | 0 / 4 | 0.0000 |
| add_client_note | 0 / 33 | 0.0000 |
| add_order_note | 0 / 14 | 0.0000 |
| add_warehouse_inventory | 0 / 18 | 0.0000 |
| adjust_financial_aid | 0 / 31 | 0.0000 |
| adjust_warehouse_quantity | 0 / 18 | 0.0000 |
| cancel_appointment | 0 / 41 | 0.0000 |
| create_claim | 0 / 5 | 0.0000 |
| create_course | 0 / 8 | 0.0000 |
| create_incident | 0 / 5 | 0.0000 |
| create_property | 0 / 20 | 0.0000 |
| create_support_ticket | 0 / 13 | 0.0000 |
| flag_fraud_investigation | 0 / 4 | 0.0000 |
| process_payout | 0 / 1 | 0.0000 |
| process_refund | 0 / 3 | 0.0000 |
| schedule_appointment | 0 / 3 | 0.0000 |
| send_carrier_message | 0 / 10 | 0.0000 |
| send_email | 0 / 8 | 0.0000 |
| send_message | 0 / 58 | 0.0000 |
| send_offer | 0 / 50 | 0.0000 |
| send_referral | 0 / 39 | 0.0000 |
| submit_customs_declaration | 0 / 9 | 0.0000 |
| submit_grade | 0 / 16 | 0.0000 |
| submit_inspection_findings | 0 / 12 | 0.0000 |
| submit_interview_feedback | 0 / 28 | 0.0000 |
| update_asset_notes | 0 / 11 | 0.0000 |
| update_candidate_status | 0 / 18 | 0.0000 |
| update_case_notes | 0 / 9 | 0.0000 |
| update_claim_description | 0 / 8 | 0.0000 |
| update_incident_description | 0 / 15 | 0.0000 |
| update_kb_article_content | 0 / 5 | 0.0000 |
| update_order_status | 0 / 21 | 0.0000 |
| update_patient_chart | 0 / 33 | 0.0000 |
| update_policy_notes | 0 / 6 | 0.0000 |
| update_product_stock | 0 / 34 | 0.0000 |
| update_property_description | 0 / 5 | 0.0000 |
| update_schedule_location | 0 / 5 | 0.0000 |
| update_shipment_status | 0 / 17 | 0.0000 |
| update_student_advisor_notes | 0 / 34 | 0.0000 |
| update_ticket_status | 0 / 4 | 0.0000 |

### indirect_sensor by source grouping


#### domain

| Value | Elevated / rows | Recall |
|---|---:|---:|
| ecommerce | 0 / 89 | 0.0000 |
| education | 0 / 93 | 0.0000 |
| healthcare | 0 / 156 | 0.0000 |
| hr | 0 / 104 | 0.0000 |
| insurance | 0 / 28 | 0.0000 |
| it_helpdesk | 0 / 36 | 0.0000 |
| legal | 0 / 14 | 0.0000 |
| logistics | 0 / 72 | 0.0000 |
| real_estate | 0 / 84 | 0.0000 |

#### attack_category

| Value | Elevated / rows | Recall |
|---|---:|---:|
| data_modification | 0 / 203 | 0.0000 |
| denial_of_service | 0 / 137 | 0.0000 |
| exfiltration | 0 / 79 | 0.0000 |
| unauthorized_action | 0 / 257 | 0.0000 |

#### injection_vector

| Value | Elevated / rows | Recall |
|---|---:|---:|
| adjuster_note_text | 0 / 4 | 0.0000 |
| advisor_notes | 0 / 22 | 0.0000 |
| application_notes | 0 / 37 | 0.0000 |
| appointment_reason | 0 / 36 | 0.0000 |
| asset_notes | 0 / 11 | 0.0000 |
| carrier_message_body | 0 / 1 | 0.0000 |
| case_notes | 0 / 6 | 0.0000 |
| case_summary | 0 / 3 | 0.0000 |
| chart_notes | 0 / 59 | 0.0000 |
| claim_description | 0 / 13 | 0.0000 |
| client_notes | 0 / 18 | 0.0000 |
| course_description | 0 / 20 | 0.0000 |
| court_location | 0 / 2 | 0.0000 |
| court_notes | 0 / 3 | 0.0000 |
| customs_additional_info | 0 / 16 | 0.0000 |
| delivery_special_instructions | 0 / 12 | 0.0000 |
| fraud_flag_reason | 0 / 4 | 0.0000 |
| grader_comments | 0 / 2 | 0.0000 |
| incident_description | 0 / 20 | 0.0000 |
| inspection_findings | 0 / 21 | 0.0000 |
| interview_feedback | 0 / 12 | 0.0000 |
| job_description | 0 / 25 | 0.0000 |
| kb_article_content | 0 / 5 | 0.0000 |
| lab_comments | 0 / 44 | 0.0000 |
| message_body | 0 / 78 | 0.0000 |
| order_notes | 0 / 34 | 0.0000 |
| payout_notes | 0 / 1 | 0.0000 |
| policy_notes | 0 / 6 | 0.0000 |
| product_description | 0 / 30 | 0.0000 |
| property_description | 0 / 23 | 0.0000 |
| resolution_notes | 0 / 11 | 0.0000 |
| resume_text | 0 / 4 | 0.0000 |
| reviewer_notes | 0 / 36 | 0.0000 |
| shipment_notes | 0 / 22 | 0.0000 |
| ticket_description | 0 / 14 | 0.0000 |
| warehouse_comments | 0 / 21 | 0.0000 |

#### target_tool

| Value | Elevated / rows | Recall |
|---|---:|---:|
| add_adjuster_note | 0 / 4 | 0.0000 |
| add_client_note | 0 / 33 | 0.0000 |
| add_order_note | 0 / 14 | 0.0000 |
| add_warehouse_inventory | 0 / 18 | 0.0000 |
| adjust_financial_aid | 0 / 31 | 0.0000 |
| adjust_warehouse_quantity | 0 / 18 | 0.0000 |
| cancel_appointment | 0 / 41 | 0.0000 |
| create_claim | 0 / 5 | 0.0000 |
| create_course | 0 / 8 | 0.0000 |
| create_incident | 0 / 5 | 0.0000 |
| create_property | 0 / 20 | 0.0000 |
| create_support_ticket | 0 / 13 | 0.0000 |
| flag_fraud_investigation | 0 / 4 | 0.0000 |
| process_payout | 0 / 1 | 0.0000 |
| process_refund | 0 / 3 | 0.0000 |
| schedule_appointment | 0 / 3 | 0.0000 |
| send_carrier_message | 0 / 10 | 0.0000 |
| send_email | 0 / 8 | 0.0000 |
| send_message | 0 / 58 | 0.0000 |
| send_offer | 0 / 50 | 0.0000 |
| send_referral | 0 / 39 | 0.0000 |
| submit_customs_declaration | 0 / 9 | 0.0000 |
| submit_grade | 0 / 16 | 0.0000 |
| submit_inspection_findings | 0 / 12 | 0.0000 |
| submit_interview_feedback | 0 / 28 | 0.0000 |
| update_asset_notes | 0 / 11 | 0.0000 |
| update_candidate_status | 0 / 18 | 0.0000 |
| update_case_notes | 0 / 9 | 0.0000 |
| update_claim_description | 0 / 8 | 0.0000 |
| update_incident_description | 0 / 15 | 0.0000 |
| update_kb_article_content | 0 / 5 | 0.0000 |
| update_order_status | 0 / 21 | 0.0000 |
| update_patient_chart | 0 / 33 | 0.0000 |
| update_policy_notes | 0 / 6 | 0.0000 |
| update_product_stock | 0 / 34 | 0.0000 |
| update_property_description | 0 / 5 | 0.0000 |
| update_schedule_location | 0 / 5 | 0.0000 |
| update_shipment_status | 0 / 17 | 0.0000 |
| update_student_advisor_notes | 0 / 34 | 0.0000 |
| update_ticket_status | 0 / 4 | 0.0000 |

### combined_untrusted_shadow by source grouping


#### domain

| Value | Elevated / rows | Recall |
|---|---:|---:|
| ecommerce | 0 / 89 | 0.0000 |
| education | 0 / 93 | 0.0000 |
| healthcare | 0 / 156 | 0.0000 |
| hr | 0 / 104 | 0.0000 |
| insurance | 0 / 28 | 0.0000 |
| it_helpdesk | 0 / 36 | 0.0000 |
| legal | 0 / 14 | 0.0000 |
| logistics | 0 / 72 | 0.0000 |
| real_estate | 0 / 84 | 0.0000 |

#### attack_category

| Value | Elevated / rows | Recall |
|---|---:|---:|
| data_modification | 0 / 203 | 0.0000 |
| denial_of_service | 0 / 137 | 0.0000 |
| exfiltration | 0 / 79 | 0.0000 |
| unauthorized_action | 0 / 257 | 0.0000 |

#### injection_vector

| Value | Elevated / rows | Recall |
|---|---:|---:|
| adjuster_note_text | 0 / 4 | 0.0000 |
| advisor_notes | 0 / 22 | 0.0000 |
| application_notes | 0 / 37 | 0.0000 |
| appointment_reason | 0 / 36 | 0.0000 |
| asset_notes | 0 / 11 | 0.0000 |
| carrier_message_body | 0 / 1 | 0.0000 |
| case_notes | 0 / 6 | 0.0000 |
| case_summary | 0 / 3 | 0.0000 |
| chart_notes | 0 / 59 | 0.0000 |
| claim_description | 0 / 13 | 0.0000 |
| client_notes | 0 / 18 | 0.0000 |
| course_description | 0 / 20 | 0.0000 |
| court_location | 0 / 2 | 0.0000 |
| court_notes | 0 / 3 | 0.0000 |
| customs_additional_info | 0 / 16 | 0.0000 |
| delivery_special_instructions | 0 / 12 | 0.0000 |
| fraud_flag_reason | 0 / 4 | 0.0000 |
| grader_comments | 0 / 2 | 0.0000 |
| incident_description | 0 / 20 | 0.0000 |
| inspection_findings | 0 / 21 | 0.0000 |
| interview_feedback | 0 / 12 | 0.0000 |
| job_description | 0 / 25 | 0.0000 |
| kb_article_content | 0 / 5 | 0.0000 |
| lab_comments | 0 / 44 | 0.0000 |
| message_body | 0 / 78 | 0.0000 |
| order_notes | 0 / 34 | 0.0000 |
| payout_notes | 0 / 1 | 0.0000 |
| policy_notes | 0 / 6 | 0.0000 |
| product_description | 0 / 30 | 0.0000 |
| property_description | 0 / 23 | 0.0000 |
| resolution_notes | 0 / 11 | 0.0000 |
| resume_text | 0 / 4 | 0.0000 |
| reviewer_notes | 0 / 36 | 0.0000 |
| shipment_notes | 0 / 22 | 0.0000 |
| ticket_description | 0 / 14 | 0.0000 |
| warehouse_comments | 0 / 21 | 0.0000 |

#### target_tool

| Value | Elevated / rows | Recall |
|---|---:|---:|
| add_adjuster_note | 0 / 4 | 0.0000 |
| add_client_note | 0 / 33 | 0.0000 |
| add_order_note | 0 / 14 | 0.0000 |
| add_warehouse_inventory | 0 / 18 | 0.0000 |
| adjust_financial_aid | 0 / 31 | 0.0000 |
| adjust_warehouse_quantity | 0 / 18 | 0.0000 |
| cancel_appointment | 0 / 41 | 0.0000 |
| create_claim | 0 / 5 | 0.0000 |
| create_course | 0 / 8 | 0.0000 |
| create_incident | 0 / 5 | 0.0000 |
| create_property | 0 / 20 | 0.0000 |
| create_support_ticket | 0 / 13 | 0.0000 |
| flag_fraud_investigation | 0 / 4 | 0.0000 |
| process_payout | 0 / 1 | 0.0000 |
| process_refund | 0 / 3 | 0.0000 |
| schedule_appointment | 0 / 3 | 0.0000 |
| send_carrier_message | 0 / 10 | 0.0000 |
| send_email | 0 / 8 | 0.0000 |
| send_message | 0 / 58 | 0.0000 |
| send_offer | 0 / 50 | 0.0000 |
| send_referral | 0 / 39 | 0.0000 |
| submit_customs_declaration | 0 / 9 | 0.0000 |
| submit_grade | 0 / 16 | 0.0000 |
| submit_inspection_findings | 0 / 12 | 0.0000 |
| submit_interview_feedback | 0 / 28 | 0.0000 |
| update_asset_notes | 0 / 11 | 0.0000 |
| update_candidate_status | 0 / 18 | 0.0000 |
| update_case_notes | 0 / 9 | 0.0000 |
| update_claim_description | 0 / 8 | 0.0000 |
| update_incident_description | 0 / 15 | 0.0000 |
| update_kb_article_content | 0 / 5 | 0.0000 |
| update_order_status | 0 / 21 | 0.0000 |
| update_patient_chart | 0 / 33 | 0.0000 |
| update_policy_notes | 0 / 6 | 0.0000 |
| update_product_stock | 0 / 34 | 0.0000 |
| update_property_description | 0 / 5 | 0.0000 |
| update_schedule_location | 0 / 5 | 0.0000 |
| update_shipment_status | 0 / 17 | 0.0000 |
| update_student_advisor_notes | 0 / 34 | 0.0000 |
| update_ticket_status | 0 / 4 | 0.0000 |

## Deterministic containment scenarios

The reference monitor commits 0/4 representative unauthorized actions. These scenarios copy only safe source IDs and categorical grouping metadata, using local synthetic canaries instead of source environment data.

## Limits

- The suite is fully synthetic, positive-only, and filtered for attacks that succeeded against the source defender.
- It has no benign controls and cannot estimate false-positive rate, precision, benign task utility, or production safety.
- Recall measures the projected injection text only, not execution of the original agent environment or deterministic trace verifier.
- The direct-user sensor is reported only as the fallback used for untrusted content, not as evidence for direct-user model fit.
- Near-overlap is heuristic; fuzzy matches are disclosed and retained as a separate evaluation suite.
- Relaxed operating points are diagnostics selected on existing validation rows, not recommendations or tuning on this source.

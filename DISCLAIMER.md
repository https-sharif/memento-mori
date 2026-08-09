# Disclaimer

**This is a research and demonstration project. It is not a medical device, and it must not be
used to make care decisions.**

Please read this before running the project, showing it to anyone, or reusing any part of it.

## Not a medical device

This software has not been evaluated, cleared, or approved by the FDA, the EMA, the MHRA, or any
other regulatory body. It is not certified under any medical-device framework. It was built as a
hackathon prototype to explore whether computer vision and language models could be combined into
a memory-support interface — nothing more.

## Not clinically validated

No part of this system has been tested in a clinical setting, reviewed by clinicians, or measured
against any standard of care. The prompts, the retrieval logic, and the behavioral categories were
written by developers, not by medical professionals.

In particular, the `POST /api/caregiver/analyze` endpoint returns a field named
`clinical_rationale`. **That name describes the shape of the data, not its authority.** It is
language-model output. It is not a clinical assessment, and no clinician reviewed it.

## The model can be wrong

The assistant is built on a large language model. Like all such models, it can:

- state things that are confidently wrong,
- invent details that were never in the patient profile,
- misread a scene, or misidentify a person or object,
- produce guidance that is inappropriate for a specific person's condition.

Face recognition additionally produces false matches and false rejections. A card naming the wrong
person is an expected failure mode of this system, not an edge case.

## Do not use this for

- Diagnosis, screening, triage, or staging of dementia or any other condition.
- Treatment, medication, or dosage decisions of any kind.
- Unsupervised care, monitoring, or companionship for a person with dementia.
- Any situation where a wrong or missing answer could affect someone's safety.
- Storing or processing real patient records, protected health information (PHI), or any data
  subject to HIPAA, GDPR, or comparable regimes.

## Always defer to professionals

Nothing this software outputs is a substitute for a qualified healthcare professional. If you are
caring for someone with dementia, decisions about their care belong with their doctor and care
team.

**In an emergency, contact your local emergency services. Do not consult this software.**

## Sample data

`patient_profile.json` describes a fictional patient ("Arthur") invented for demonstration. It is
not a real person's record, and the repository contains no real patient data.

## Liability

This project is provided under the MIT License, which includes no warranty of any kind. See
[LICENSE](LICENSE). The authors accept no liability for any use of this software.

See also [PRIVACY.md](PRIVACY.md) for how the system handles biometric data.

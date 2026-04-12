"""ProjectManager mixins grouped by operational responsibility.

Each mixin owns a cohesive slice of behavior (state I/O, runtime state,
chunk CRUD, generation, ASR proofread, repair, export, and voice handling).
``project.py`` composes them into the final ``ProjectManager`` class.
"""

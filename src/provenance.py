import ast
import hashlib
import inspect
import textwrap


STUDY_PROTOCOL_VERSION = "audit-v3"


def implementation_signature(*functions):
    # Сравниваем структуру кода, а не текст файла. Комментарии и перенос в notebook
    # не меняют подпись; изменение вычислений должно остановить reuse старого run.
    bodies = [ast.dump(ast.parse(textwrap.dedent(inspect.getsource(f))), include_attributes=False)
              for f in functions]
    return hashlib.sha256("\n".join(bodies).encode()).hexdigest()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeCloak — Universal Obfuscator (Python / PowerShell / Batch)
- GUI (tkinter) with branded logo + modern dark theme
- CLI (argparse)
- Python: scope-aware AST rename + multi-level obfuscation (1..6)
- PowerShell (.ps1) and Batch (.bat/.cmd) obfuscation (basic but useful)
- Logging: .log file with SHA256, time, level (brand references)
- Test mode: execute original and obfuscated code (use with caution)
Author: adapted for user as "CodeCloak"
"""

from __future__ import annotations
import argparse
import ast
import base64
import codecs
import copy
import hashlib
import io
import json
import os
import random
import re
import shlex
import string
import subprocess
import sys
import textwrap
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# Optional: zlib for compression layers
try:
    import zlib
except Exception:
    zlib = None

# -----------------------------
# Small config / branding
# -----------------------------
APP_NAME = "CodeCloak"
VERSION = "1.0"

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())

def rand_name(min_len=5, max_len=10):
    return ''.join(random.choice(string.ascii_letters) for _ in range(random.randint(min_len, max_len)))

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_text(path: str, s: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

# -----------------------------
# File type detection
# -----------------------------
def detect_file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        return "python"
    if ext == ".ps1":
        return "powershell"
    if ext in (".bat", ".cmd"):
        return "batch"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
            if first.startswith("#!") and "python" in first.lower():
                return "python"
    except Exception:
        pass
    return "text"

# -----------------------------
# Python obfuscation utilities
# -----------------------------
PY_KEYWORDS = set([
    'False','None','True','and','as','assert','async','await','break','class','continue',
    'def','del','elif','else','except','finally','for','from','global','if','import','in',
    'is','lambda','nonlocal','not','or','pass','raise','return','try','while','with','yield'
])
PY_BUILTINS = set(dir(__builtins__))

class ScopeAwareRenamer(ast.NodeTransformer):
    """Scope-aware AST renamer — safer renaming respecting scope/global declarations"""
    def __init__(self, protect=None):
        super().__init__()
        self.scopes = [{}]  # stack of dicts: original->new
        self.global_decls = [set()]
        self.protect = set(protect or [])

    def _enter(self):
        self.scopes.append({})
        self.global_decls.append(set())

    def _exit(self):
        self.scopes.pop()
        self.global_decls.pop()

    def _find(self, name):
        for s in reversed(self.scopes):
            if name in s:
                return s[name]
        return None

    def _assign(self, name, idx=None):
        if idx is None:
            idx = len(self.scopes) - 1
        if name in self.scopes[idx]:
            return self.scopes[idx][name]
        if name in PY_KEYWORDS or name in PY_BUILTINS or name in self.protect:
            return name
        if name.startswith("__") and name.endswith("__"):
            return name
        new = rand_name()
        while any(new in s.values() for s in self.scopes) or new in PY_KEYWORDS or new in PY_BUILTINS:
            new = rand_name()
        self.scopes[idx][name] = new
        return new

    def visit_FunctionDef(self, node):
        # map function name in parent scope or module if declared global
        if node.name not in self.scopes[-1]:
            if node.name in self.global_decls[-1]:
                self._assign(node.name, 0)
            else:
                self._assign(node.name)
        node.name = self._find(node.name) or node.name
        # enter function scope
        self._enter()
        # rename args
        for a in node.args.args:
            if a.arg:
                a.arg = self._assign(a.arg)
        if node.args.vararg:
            node.args.vararg.arg = self._assign(node.args.vararg.arg)
        if node.args.kwarg:
            node.args.kwarg.arg = self._assign(node.args.kwarg.arg)
        for a in node.args.kwonlyargs:
            a.arg = self._assign(a.arg)
        self.generic_visit(node)
        self._exit()
        return node

    def visit_ClassDef(self, node):
        if node.name not in self.scopes[-1]:
            self._assign(node.name)
        node.name = self._find(node.name) or node.name
        self._enter()
        self.generic_visit(node)
        self._exit()
        return node

    def visit_Global(self, node):
        for n in node.names:
            self.global_decls[-1].add(n)
            if n not in self.scopes[0]:
                self._assign(n, 0)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            if node.id in self.global_decls[-1]:
                idx = 0
            else:
                idx = len(self.scopes)-1
            node.id = self._assign(node.id, idx)
        elif isinstance(node.ctx, ast.Load):
            mapped = self._find(node.id)
            if mapped:
                node.id = mapped
        elif isinstance(node.ctx, ast.Del):
            mapped = self._find(node.id)
            if mapped:
                node.id = mapped
        return node

def py_minify(src: str) -> str:
    src = re.sub(r'(?m)#.*$', '', src)
    lines = [ln.strip() for ln in src.splitlines()]
    lines = [ln for ln in lines if ln]
    return ';'.join(lines)

def py_replace_strings_b64(src: str) -> str:
    def repl(m):
        s = m.group(0)
        inner = s[1:-1]
        b = base64.b64encode(inner.encode('utf-8')).decode('ascii')
        return f"base64.b64decode('{b}').decode()"
    pattern = r'\'(?:[^\\\']|\\.)*\'|"(?:[^\\"]|\\.)*"'
    return re.sub(pattern, repl, src)

class PerFunctionEncryptor(ast.NodeTransformer):
    def __init__(self, use_zlib=True):
        super().__init__()
        self.use_zlib = use_zlib and (zlib is not None)
        self.counter = 0
        self.additions = []

    def visit_FunctionDef(self, node):
        try:
            src = ast.unparse(node)
        except Exception:
            src = f"def {node.name}(*args, **kwargs):\n    pass"
        self.counter += 1
        tag = f"f_enc_{self.counter}"
        if self.use_zlib:
            comp = zlib.compress(src.encode('utf-8'))
            b64 = base64.b64encode(comp).decode('ascii')
            rot = codecs.encode(b64, 'rot_13')
            decode = f"_src_{tag} = zlib.decompress(base64.b64decode(codecs.decode('{rot}','rot_13'))).decode()"
        else:
            b64 = base64.b64encode(src.encode('utf-8')).decode('ascii')
            rot = codecs.encode(b64, 'rot_13')
            decode = f"_src_{tag} = base64.b64decode(codecs.decode('{rot}','rot_13')).decode()"
        wrapper = textwrap.dedent(f"""
            # CodeCloak encrypted wrapper for {node.name}
            {decode}
            _loaded_{tag} = False
            def {node.name}(*args, **kwargs):
                global _loaded_{tag}
                if not _loaded_{tag}:
                    exec(_src_{tag}, globals())
                    _loaded_{tag} = True
                return globals()['{node.name}'](*args, **kwargs)
        """)
        placeholder = ast.FunctionDef(name=node.name, args=node.args, body=[ast.Pass()], decorator_list=[], returns=node.returns)
        self.additions.append(wrapper)
        return placeholder

def obfuscate_python(src: str, level: int) -> str:
    orig = src
    src = src.replace('\r\n', '\n').replace('\r', '\n')
    # remove docstrings
    try:
        tree = ast.parse(src)
        class DocRem(ast.NodeTransformer):
            def visit_Module(self, node):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant, ast.Str)):
                    node.body.pop(0)
                self.generic_visit(node)
                return node
            def visit_FunctionDef(self, node):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant, ast.Str)):
                    node.body.pop(0)
                self.generic_visit(node)
                return node
            def visit_ClassDef(self, node):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant, ast.Str)):
                    node.body.pop(0)
                self.generic_visit(node)
                return node
        tree = DocRem().visit(tree)
        ast.fix_missing_locations(tree)
        src = ast.unparse(tree)
    except Exception:
        src = re.sub(r'("""|\'\'\')[\s\S]*?\1', '', src)

    if level >= 1:
        try:
            tree = ast.parse(src)
            ren = ScopeAwareRenamer()
            tree = ren.visit(tree)
            ast.fix_missing_locations(tree)
            src = ast.unparse(tree)
        except Exception:
            pass

    if level >= 2:
        lines = src.splitlines()
        out = []
        for ln in lines:
            if random.random() < 0.22:
                out.append("# " + rand_name(10))
            out.append(ln)
            if random.random() < 0.18:
                out.append(f"{rand_name(6)} = {random.randint(0,9999)}")
        if len(out) > 6:
            blocks = [out[i:i+3] for i in range(0, len(out), 3)]
            random.shuffle(blocks)
            out = [l for b in blocks for l in b]
        src = "\n".join(out)

    if level >= 3:
        src = py_replace_strings_b64(src)
        if "import base64" not in src:
            src = "import base64\n" + src

    if level >= 4:
        payload = src
        if zlib is not None:
            comp = zlib.compress(payload.encode('utf-8'))
            b64 = base64.b64encode(comp).decode('ascii')
            wrapper = textwrap.dedent(f"""
                import base64,zlib
                exec(compile(zlib.decompress(base64.b64decode('{b64}')).decode(),'<obf>','exec'))
            """)
        else:
            b64 = base64.b64encode(payload.encode('utf-8')).decode('ascii')
            wrapper = textwrap.dedent(f"""
                import base64
                exec(compile(base64.b64decode('{b64}').decode(),'<obf>','exec'))
            """)
        src = wrapper

    if level >= 5:
        minified = py_minify(src)
        if zlib is not None:
            comp = zlib.compress(minified.encode('utf-8'))
            b64 = base64.b64encode(comp).decode('ascii')
            rot = codecs.encode(b64, 'rot_13')
            final = textwrap.dedent(f"""
                import base64,zlib,codecs
                _d = codecs.decode('{rot}','rot_13')
                _s = zlib.decompress(base64.b64decode(_d)).decode()
                eval(compile(_s,'<obf>','exec'))
            """)
        else:
            b64 = base64.b64encode(minified.encode('utf-8')).decode('ascii')
            rot = codecs.encode(b64, 'rot_13')
            final = textwrap.dedent(f"""
                import base64,codecs
                _d = codecs.decode('{rot}','rot_13')
                _s = base64.b64decode(_d).decode()
                eval(compile(_s,'<obf>','exec'))
            """)
        src = final

    if level == 6:
        try:
            mod = ast.parse(orig)
            enc = PerFunctionEncryptor(use_zlib=(zlib is not None))
            mod2 = enc.visit(copy.deepcopy(mod))
            ast.fix_missing_locations(mod2)
            try:
                transformed = ast.unparse(mod2)
            except Exception:
                transformed = orig
            additions = "\n\n".join(enc.additions)
            header = ""
            if "import base64" not in transformed and "base64.b64decode" in additions:
                header += "import base64\n"
            if "codecs" not in transformed and "codecs.decode" in additions:
                header += "import codecs\n"
            if zlib is not None and "zlib" not in transformed and "zlib.decompress" in additions:
                header += "import zlib\n"
            src = header + transformed + "\n\n# CodeCloak encrypted wrappers\n" + additions
        except Exception:
            pass

    return src

# -----------------------------
# PowerShell obfuscation (basic)
# -----------------------------
_ps_var_pattern = re.compile(r'(?<![A-Za-z0-9_])\$[A-Za-z_][A-Za-z0-9_]*')

def ps_find_vars(src: str):
    return sorted(set(m.group(0) for m in _ps_var_pattern.finditer(src)))

def ps_replace_vars(src: str, mapping: dict):
    def repl(m):
        v = m.group(0)
        return mapping.get(v, v)
    return _ps_var_pattern.sub(repl, src)

def ps_encode_strings_b64_unicode(src: str):
    pattern = r'(?s)(\'(?:[^\\\']|\\.)*\'|"(?:[^\\"]|\\.)*")'
    def repl(m):
        s = m.group(0)
        inner = s[1:-1]
        b = inner.encode('utf-16le')
        b64 = base64.b64encode(b).decode('ascii')
        expr = f"[System.Text.Encoding]::Unicode.GetString([System.Convert]::FromBase64String('{b64}'))"
        return expr
    return re.sub(pattern, repl, src)

def obfuscate_powershell(src: str, level: int):
    orig = src
    if level >= 1:
        vars_found = ps_find_vars(src)
        mapping = {}
        for v in vars_found:
            if ':' in v:
                continue
            new = '$' + rand_name(6,8)
            while new in mapping.values():
                new = '$' + rand_name(6,8)
            mapping[v] = new
        if mapping:
            src = ps_replace_vars(src, mapping)
    if level >= 2:
        lines = src.splitlines()
        out = []
        for ln in lines:
            if random.random() < 0.25:
                out.append("<!-- " + rand_name(8) + " -->")
            out.append(ln)
            if random.random() < 0.18:
                out.append("Write-Verbose '" + rand_name(6) + "'")
        src = "\n".join(out)
    if level >= 3:
        try:
            src = ps_encode_strings_b64_unicode(src)
        except Exception:
            pass
    if level >= 4:
        payload = src
        b64 = base64.b64encode(payload.encode('utf-8')).decode('ascii')
        decode = f"$s = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{b64}'))"
        wrapper = decode + "\nInvoke-Expression $s"
        src = wrapper
    return src

# -----------------------------
# Batch obfuscation (basic)
# -----------------------------
_bat_set_re = re.compile(r'(?mi)^\s*set\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$')
_bat_var_use_re = re.compile(r'%([A-Za-z_][A-Za-z0-9_]*)%')

def bat_find_set_vars(src: str):
    return set(m.group(1) for m in _bat_set_re.finditer(src))

def bat_rename_vars(src: str, mapping: dict):
    def repl(m):
        name = m.group(1)
        val = m.group(2)
        return f"set {mapping.get(name,name)}={val}"
    out = _bat_set_re.sub(repl, src)
    out = _bat_var_use_re.sub(lambda m: "%" + mapping.get(m.group(1), m.group(1)) + "%", out)
    return out

def obfuscate_batch(src: str, level: int):
    orig = src
    if level >= 1:
        vars_found = bat_find_set_vars(src)
        mapping = {}
        for v in vars_found:
            new = rand_name(4,8).upper()
            while new in mapping.values():
                new = rand_name(4,8).upper()
            mapping[v] = new
        if mapping:
            src = bat_rename_vars(src, mapping)
    if level >= 2:
        lines = src.splitlines()
        out = []
        for ln in lines:
            if random.random() < 0.22:
                out.append("REM " + rand_name(8))
            out.append(ln)
            if random.random() < 0.12:
                out.append("echo " + rand_name(6) + " >nul")
        src = "\n".join(out)
    if level >= 3:
        lines = src.splitlines()
        if len(lines) > 4:
            new_lines = []
            label_ctr = 0
            i = 0
            while i < len(lines):
                if random.random() < 0.15:
                    label_ctr += 1
                    new_lines.append(f"goto LBL{label_ctr}")
                    new_lines.append(f":LBL{label_ctr}")
                    for j in range(i, min(i+2, len(lines))):
                        new_lines.append(lines[j])
                    i += 2
                else:
                    new_lines.append(lines[i])
                    i += 1
            src = "\n".join(new_lines)
    return src

# -----------------------------
# Generic facade
# -----------------------------
def obfuscate_any(path: str, level: int):
    typ = detect_file_type(path)
    src = read_text(path)
    if typ == "python":
        return obfuscate_python(src, level), "python"
    if typ == "powershell":
        return obfuscate_powershell(src, level), "powershell"
    if typ == "batch":
        return obfuscate_batch(src, level), "batch"
    # fallback: add simple header
    if level >= 1:
        header = f":: {APP_NAME} obfuscation header\n"
        return header + src, "text"
    return src, "text"

# -----------------------------
# Test runners
# -----------------------------
def run_python_capture(code: str):
    g = {"__name__":"__main__"}
    old_stdout = sys.stdout
    buf = io.StringIO()
    try:
        sys.stdout = buf
        exec(compile(code, "<obf>", "exec"), g)
        exc = None
    except Exception as e:
        exc = {"type": type(e).__name__, "value": str(e), "traceback": traceback.format_exc()}
    finally:
        sys.stdout = old_stdout
    return buf.getvalue(), exc, g

def run_external_capture_file(path: str, timeout=8):
    typ = detect_file_type(path)
    if typ == "powershell":
        tries = [["pwsh","-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File", path],
                 ["powershell","-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File", path]]
    elif typ == "batch":
        tries = [["cmd","/c", path]]
    else:
        tries = [[sys.executable, path]]
    for cmd in tries:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return proc.stdout, (proc.returncode != 0 and proc.stderr or None)
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return "", {"type":"Timeout","value":"timeout"}
    return "", {"type":"Runtime","value":"no-runner-found"}

def compare_and_test(original_path: str, obf_text: str, obf_type: str, out_path: str, check_vars: list|None=False):
    orig_src = read_text(original_path)
    result = {"success": False, "orig_stdout":"", "obf_stdout":"", "orig_exc":None, "obf_exc":None, "vars_equal":None, "var_diffs":{}}
    if obf_type == "python":
        orig_out, orig_exc, orig_g = run_python_capture(orig_src)
        obf_out, obf_exc, obf_g = run_python_capture(obf_text)
        result["orig_stdout"] = orig_out
        result["obf_stdout"] = obf_out
        result["orig_exc"] = orig_exc
        result["obf_exc"] = obf_exc
        stdout_match = (orig_out == obf_out)
        result["success"] = (orig_exc is None and obf_exc is None and stdout_match)
        if check_vars:
            result["vars_equal"] = True
            for v in check_vars:
                o = orig_g.get(v, "<MISSING>")
                b = obf_g.get(v, "<MISSING>")
                if o != b:
                    result["vars_equal"] = False
                    result["var_diffs"][v] = {"orig":repr(o), "obf":repr(b)}
            if not result["vars_equal"]:
                result["success"] = False
    else:
        tmp = out_path
        try:
            write_text(tmp, obf_text)
            orig_out, orig_err = run_external_capture_file(original_path)
            obf_out, obf_err = run_external_capture_file(tmp)
            result["orig_stdout"] = orig_out
            result["obf_stdout"] = obf_out
            result["orig_exc"] = orig_err
            result["obf_exc"] = obf_err
            result["success"] = (orig_err is None and obf_err is None and orig_out == obf_out)
        except Exception as e:
            result["obf_exc"] = {"type":"Exception","value":str(e)}
    return result

# -----------------------------
# Logging
# -----------------------------
def write_log(input_path: str, output_path: str, level: int, test_result=None):
    try:
        raw = read_text(input_path).encode('utf-8', errors='ignore')
        data = {
            "app": APP_NAME,
            "version": VERSION,
            "input_file": os.path.abspath(input_path),
            "input_sha256": sha256_bytes(raw),
            "time": now_iso(),
            "level": level,
            "output_file": os.path.abspath(output_path),
            "test": None
        }
        if test_result is not None:
            data["test"] = {
                "success": bool(test_result.get("success")),
                "orig_stdout_preview": (test_result.get("orig_stdout") or "")[:200],
                "obf_stdout_preview": (test_result.get("obf_stdout") or "")[:200]
            }
        log_path = output_path + ".log"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return log_path
    except Exception:
        return None

# -----------------------------
# CLI
# -----------------------------
def cli_main():
    parser = argparse.ArgumentParser(prog=APP_NAME, description=f"{APP_NAME} — universal obfuscator (py/ps1/bat)")
    parser.add_argument("--input","-i", required=True, help="Input file")
    parser.add_argument("--output","-o", help="Output file (default: input.obf_l{level}.ext)")
    parser.add_argument("--level","-l", type=int, choices=[1,2,3,4,5,6], default=2, help="Obfuscation level")
    parser.add_argument("--test", action="store_true", help="Run test after obfuscation")
    parser.add_argument("--check-vars", nargs="*", help="For python: names of globals to compare")
    args = parser.parse_args()

    inp = args.input
    if not os.path.isfile(inp):
        print(f"Error: input '{inp}' not found.")
        sys.exit(2)
    level = args.level
    obf_text, obf_type = obfuscate_any(inp, level)
    out_path = args.output or (os.path.splitext(inp)[0] + f".obf_l{level}" + os.path.splitext(inp)[1])
    write_text(out_path, obf_text)
    print(f"[{APP_NAME}] Wrote obfuscated ({obf_type}) to {out_path}")

    test_result = None
    if args.test:
        print(f"[{APP_NAME}] Running test (executing original and obfuscated)...")
        test_result = compare_and_test(inp, obf_text, obf_type, out_path, check_vars=args.check_vars)
        print("[Test] success:", test_result["success"])
    logp = write_log(inp, out_path, level, test_result=test_result)
    if logp:
        print(f"[{APP_NAME}] Log saved to {logp}")

# -----------------------------
# GUI: draw logo and UI (CodeCloak branded)
# -----------------------------
class Style:
    BG = "#0b0f0a"
    PANEL = "#07120f"
    ACCENT = "#00ff9c"
    ACCENT2 = "#00d1ff"
    TXT = "#b7f5d8"
    MUTED = "#6fbfa0"
    BTN_BG = "#062016"
    BTN_FG = "#b7f5d8"
    FONT = ("Consolas", 11)

def draw_logo(canvas: tk.Canvas, x=10, y=10, w=220, h=60):
    """
    Draws a simple programmatic 'CodeCloak' logo on the canvas — shield + text.
    No external images required.
    """
    # shield shape (rounded triangle)
    cx = x + 40
    cy = y + h//2
    # shield outer
    canvas.create_polygon(
        cx, y+6,
        x+10, y+18,
        x+10, y+h-12,
        cx, y+h,
        x+70, y+h-12,
        x+70, y+18,
        fill="#002218", outline=Style.ACCENT2, width=2
    )
    # inner cloak arc
    canvas.create_arc(x+8, y+6, x+72, y+h+6, start=200, extent=140, style='arc', outline=Style.ACCENT, width=3)
    # key text
    canvas.create_text(x+110, y+16, anchor="nw", text=APP_NAME, font=("Consolas", 18, "bold"), fill=Style.ACCENT)
    canvas.create_text(x+110, y+36, anchor="nw", text="Advanced Script Obfuscator", font=("Consolas", 10), fill=Style.MUTED)

class CodeCloakGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"{APP_NAME} — v{VERSION}")
        root.configure(bg=Style.BG)
        root.geometry("980x720")

        # header
        header = tk.Frame(root, bg=Style.PANEL, pady=8)
        header.pack(fill=tk.X, padx=10, pady=8)
        canv = tk.Canvas(header, width=420, height=70, bg=Style.PANEL, highlightthickness=0)
        canv.pack(side=tk.LEFT, padx=6)
        draw_logo(canv, x=6, y=6, w=420, h=60)

        info = tk.Frame(header, bg=Style.PANEL)
        info.pack(side=tk.LEFT, padx=6)
        tk.Label(info, text=APP_NAME, bg=Style.PANEL, fg=Style.ACCENT, font=("Consolas", 18, "bold")).pack(anchor="w")
        tk.Label(info, text=f"Version {VERSION} — Universal obfuscator (py/ps1/bat)", bg=Style.PANEL, fg=Style.MUTED, font=("Consolas", 10)).pack(anchor="w")

        # controls
        ctrl = tk.Frame(root, bg=Style.BG)
        ctrl.pack(fill=tk.X, padx=10, pady=6)
        tk.Label(ctrl, text="File:", bg=Style.BG, fg=Style.ACCENT2, font=Style.FONT).pack(side=tk.LEFT)
        self.entry_file = tk.Entry(ctrl, width=70, bg="#00100a", fg=Style.TXT, insertbackground=Style.TXT, font=Style.FONT)
        self.entry_file.pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="Choose", command=self.choose_file, bg=Style.BTN_BG, fg=Style.BTN_FG).pack(side=tk.LEFT, padx=6)

        # level & options
        opt = tk.Frame(root, bg=Style.BG)
        opt.pack(fill=tk.X, padx=10, pady=6)
        tk.Label(opt, text="Level:", bg=Style.BG, fg=Style.ACCENT2, font=Style.FONT).pack(side=tk.LEFT)
        self.level_var = tk.IntVar(value=2)
        for i in range(1,7):
            tk.Radiobutton(opt, text=str(i), variable=self.level_var, value=i, bg=Style.BG, fg=Style.TXT, selectcolor="#002218", font=Style.FONT).pack(side=tk.LEFT, padx=3)
        self.test_var = tk.IntVar(value=0)
        tk.Checkbutton(opt, text="Run test (exec)", variable=self.test_var, bg=Style.BG, fg=Style.MUTED, selectcolor="#002218", font=("Consolas",10)).pack(side=tk.LEFT, padx=8)
        tk.Label(opt, text="Check globals (py):", bg=Style.BG, fg=Style.MUTED, font=("Consolas",10)).pack(side=tk.LEFT, padx=6)
        self.entry_check = tk.Entry(opt, width=30, bg="#00100a", fg=Style.TXT, insertbackground=Style.TXT, font=Style.FONT)
        self.entry_check.pack(side=tk.LEFT, padx=6)

        # ops
        ops = tk.Frame(root, bg=Style.BG)
        ops.pack(fill=tk.X, padx=10, pady=6)
        tk.Button(ops, text="Obfuscate", command=self.action_obfuscate, bg="#003126", fg=Style.BTN_FG, font=("Consolas",11,"bold")).pack(side=tk.LEFT, padx=6)
        tk.Button(ops, text="Save", command=self.action_save, bg="#002437", fg=Style.BTN_FG).pack(side=tk.LEFT, padx=6)
        tk.Button(ops, text="Copy", command=self.action_copy, bg="#002437", fg=Style.BTN_FG).pack(side=tk.LEFT, padx=6)
        tk.Button(ops, text="Clear", command=self.action_clear, bg="#222", fg=Style.MUTED).pack(side=tk.LEFT, padx=6)

        # preview
        pv = tk.Frame(root, bg=Style.BG)
        pv.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        tk.Label(pv, text="Preview:", bg=Style.BG, fg=Style.ACCENT, font=("Consolas",12,"bold")).pack(anchor="w")
        self.txt = scrolledtext.ScrolledText(pv, bg="#001210", fg=Style.TXT, insertbackground=Style.TXT, font=("Consolas",11))
        self.txt.pack(fill=tk.BOTH, expand=True)

        # status
        status = tk.Frame(root, bg=Style.PANEL)
        status.pack(fill=tk.X, padx=10, pady=6)
        self.lbl_status = tk.Label(status, text=f"{APP_NAME} ready", bg=Style.PANEL, fg=Style.MUTED, font=("Consolas",10))
        self.lbl_status.pack(side=tk.LEFT, padx=6)

        # internal state
        self.current_text = None
        self.current_type = None
        self.current_input = None

    def choose_file(self):
        p = filedialog.askopenfilename(filetypes=[("Scripts","*.py *.ps1 *.bat *.cmd"),("All","*.*")])
        if p:
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, p)

    def action_obfuscate(self):
        path = self.entry_file.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror(APP_NAME, "Choose a valid file first.")
            return
        level = self.level_var.get()
        self.lbl_status.config(text=f"{APP_NAME}: obfuscating {os.path.basename(path)} (L{level})...")
        self.root.update()
        obf_text, typ = obfuscate_any(path, level)
        self.current_text = obf_text
        self.current_type = typ
        self.current_input = path
        self.show_preview(obf_text)
        self.lbl_status.config(text=f"{APP_NAME}: obfuscated ({typ}). Ready to save.")
        # optional test
        if self.test_var.get():
            self.lbl_status.config(text=f"{APP_NAME}: running test (this executes code!)")
            self.root.update()
            check_vars = self.entry_check.get().split() if self.entry_check.get().strip() else None
            out_path = os.path.splitext(path)[0] + f".obf_l{level}" + os.path.splitext(path)[1]
            res = compare_and_test(path, obf_text, typ, out_path, check_vars)
            if res["success"]:
                messagebox.showinfo(APP_NAME + " — Test OK", "Original and obfuscated behaved similarly.")
                self.lbl_status.config(text=f"{APP_NAME}: test OK")
            else:
                messagebox.showwarning(APP_NAME + " — Test FAILED", "Behavior differs — check preview and logs.")
                self.lbl_status.config(text=f"{APP_NAME}: test failed")

    def show_preview(self, s: str):
        self.txt.delete(1.0, tk.END)
        header = f"--- {APP_NAME} preview ({self.current_type}) ---\n"
        self.txt.insert(tk.END, header)
        self.txt.insert(tk.END, s)

    def action_save(self):
        if not self.current_text:
            messagebox.showerror(APP_NAME, "No obfuscated code to save. Obfuscate first.")
            return
        default = os.path.splitext(self.current_input)[0] + f".obf_l{self.level_var.get()}" + os.path.splitext(self.current_input)[1]
        p = filedialog.asksaveasfilename(defaultextension=os.path.splitext(default)[1], initialfile=os.path.basename(default))
        if not p:
            return
        write_text(p, self.current_text)
        logp = write_log(self.current_input, p, self.level_var.get(), test_result=None)
        messagebox.showinfo(APP_NAME, f"Saved to:\n{p}\nLog: {logp}")

    def action_copy(self):
        if not self.current_text:
            messagebox.showerror(APP_NAME, "No obfuscated code to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.current_text)
        messagebox.showinfo(APP_NAME, "Copied to clipboard.")

    def action_clear(self):
        self.txt.delete(1.0, tk.END)
        self.current_text = None
        self.current_input = None
        self.lbl_status.config(text=f"{APP_NAME} ready")

# -----------------------------
# main
# -----------------------------
def main():
    if len(sys.argv) > 1:
        cli_main()
    else:
        root = tk.Tk()
        gui = CodeCloakGUI(root)
        root.mainloop()

if __name__ == "__main__":
    main()

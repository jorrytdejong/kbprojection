import os
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import Mock, patch

from kbprojection.easyccg_vendor import _find_easyccg_model_dir, install_local_easyccg
from kbprojection.loaders.sick import SICKLoader
from kbprojection.loaders.snli import SNLILoader
from kbprojection.settings import (
    DEFAULT_EASYCCG_MODEL_SOURCE,
    DEFAULT_EASYCCG_REPO,
    DEFAULT_LANGPRO_REPO,
    DEFAULT_LANGPRO_ENDPOINT,
    format_local_langpro_missing_error,
    enable_local,
    get_app_dir,
    get_default_cache_dir,
    get_default_dataset_dir,
    get_default_easyccg_vendor_dir,
    get_default_results_dir,
    get_default_vendor_dir,
    get_langpro_settings,
    resolve_local_langpro_root,
)


class TestSettingsPaths(unittest.TestCase):
    def test_app_dir_uses_localappdata_on_windows(self):
        with patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\Test\AppData\Local"}, clear=True):
            # Model Windows path semantics without changing the host OS globally.
            with patch("kbprojection.settings.os", Mock(name="nt", environ=os.environ)) as windows_os, \
                 patch("kbprojection.settings.Path", PureWindowsPath):
                windows_os.name = "nt"
                self.assertEqual(
                    get_app_dir(),
                    PureWindowsPath(r"C:\Users\Test\AppData\Local") / "kbprojection",
                )

    def test_app_dir_override_wins(self):
        with patch.dict(os.environ, {"KBPROJECTION_APP_DIR": r"C:\custom\kbprojection"}, clear=True):
            self.assertEqual(get_app_dir(), Path(r"C:\custom\kbprojection"))

    def test_default_roots_respect_env_overrides(self):
        env = {
            "KBPROJECTION_APP_DIR": r"C:\app",
            "KBPROJECTION_DATA_DIR": r"C:\data",
            "KBPROJECTION_RESULTS_DIR": r"C:\results",
            "KBPROJECTION_CACHE_DIR": r"C:\cache",
            "KBPROJECTION_LANGPRO_VENDOR_DIR": r"C:\vendor\LangPro",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_default_dataset_dir("sick"), Path(r"C:\data") / "sick")
            self.assertEqual(get_default_results_dir(), Path(r"C:\results"))
            self.assertEqual(get_default_cache_dir(), Path(r"C:\cache"))
            self.assertEqual(get_default_vendor_dir(), Path(r"C:\vendor\LangPro"))

    def test_langpro_default_cache_path_uses_app_dir(self):
        with patch.dict(os.environ, {"KBPROJECTION_APP_DIR": r"C:\app"}, clear=True):
            settings = get_langpro_settings()
            self.assertEqual(settings.cache_path, Path(r"C:\app") / "langpro_cache.sqlite3")

    def test_dataset_loaders_default_to_app_data_dataset_dirs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(os.environ, {"KBPROJECTION_DATA_DIR": tmp_dir}, clear=True):
                self.assertEqual(SICKLoader().data_dir, Path(tmp_dir) / "sick")
                self.assertEqual(SNLILoader().data_dir, Path(tmp_dir) / "snli")

    def test_dataset_loaders_keep_explicit_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            explicit = Path(tmp_dir) / "custom"
            self.assertEqual(SICKLoader(data_dir=explicit).data_dir, explicit)
            self.assertEqual(SNLILoader(data_dir=explicit).data_dir, explicit)

    def test_default_vendor_dir_uses_app_dir(self):
        with patch.dict(os.environ, {"KBPROJECTION_APP_DIR": r"C:\app"}, clear=True):
            self.assertEqual(get_default_vendor_dir(), Path(r"C:\app") / "vendor" / "LangPro")
            self.assertEqual(get_default_easyccg_vendor_dir(), Path(r"C:\app") / "vendor" / "easyccg")

    def test_enable_local_sets_local_langpro_environment(self):
        env = {
            "KBPROJECTION_APP_DIR": r"C:\app",
            "KBPROJECTION_LANGPRO_ENDPOINT": DEFAULT_LANGPRO_ENDPOINT,
        }
        with patch.dict(os.environ, env, clear=True):
            settings = enable_local(
                local_root=r"C:\LangPro",
                easyccg_dir=r"C:\easyccg",
                easyccg_model_source="https://example.test/model.tar.gz",
                auto_clone=True,
                swipl="custom-swipl",
            )

        self.assertEqual(settings.endpoint, "local://auto")
        self.assertEqual(settings.local_root, Path(r"C:\LangPro"))
        self.assertEqual(settings.local_easyccg_dir, Path(r"C:\easyccg"))
        self.assertTrue(settings.local_auto_clone)
        self.assertEqual(settings.local_swipl, "custom-swipl")
        self.assertEqual(settings.local_easyccg_model_source, "https://example.test/model.tar.gz")
        self.assertTrue(DEFAULT_EASYCCG_MODEL_SOURCE.endswith("easyccg-model-rebank.tar.gz"))
        self.assertEqual(DEFAULT_EASYCCG_REPO, "https://github.com/mikelewis0/easyccg.git")

    @patch("kbprojection.easyccg_vendor._download_or_unpack_model")
    @patch("kbprojection.easyccg_vendor._run_git")
    def test_install_local_easyccg_uses_app_vendor_dir(self, mock_run_git, mock_download_model):
        mock_run_git.return_value = Mock(returncode=0, stdout="", stderr="")
        def create_model(source, target):
            target.mkdir(parents=True, exist_ok=True)
            (target / "easyccg.jar").write_text("jar", encoding="utf-8")
            (target / "model_rebank").mkdir(parents=True)

        mock_download_model.side_effect = create_model
        with tempfile.TemporaryDirectory() as tmp_dir:
            app_dir = Path(tmp_dir) / "app"
            with patch.dict(os.environ, {"KBPROJECTION_APP_DIR": str(app_dir)}, clear=True):
                target = install_local_easyccg()

            self.assertEqual(target, app_dir / "vendor" / "easyccg")
            mock_run_git.assert_called_once_with(
                ["clone", "--depth", "1", DEFAULT_EASYCCG_REPO, str(target)]
            )
            mock_download_model.assert_called_once_with(DEFAULT_EASYCCG_MODEL_SOURCE, target)

    def test_easyccg_model_detection_accepts_rebank_archive_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "easyccg-model-rebank"
            model_dir.mkdir()
            for name in ("bias", "binaryRules", "categories", "classifier", "unaryRules"):
                (model_dir / name).write_text("", encoding="utf-8")

            self.assertEqual(_find_easyccg_model_dir(Path(tmp_dir)), model_dir)


class TestLocalLangProDiscovery(unittest.TestCase):
    def test_env_root_wins_even_if_missing(self):
        with patch.dict(os.environ, {"KBPROJECTION_LANGPRO_LOCAL_ROOT": r"C:\LangPro"}, clear=True):
            self.assertEqual(resolve_local_langpro_root(), Path(r"C:\LangPro"))

    def test_repo_vendor_wins_over_sibling_and_app_vendor(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "repo"
            repo_vendor = project_root / "vendor" / "LangPro"
            sibling = project_root.parent / "LangPro"
            app_vendor = Path(tmp_dir) / "app" / "vendor" / "LangPro"
            for path in (repo_vendor, sibling, app_vendor):
                path.mkdir(parents=True)

            with patch("kbprojection.settings.PROJECT_ROOT", project_root):
                with patch.dict(os.environ, {"KBPROJECTION_APP_DIR": str(Path(tmp_dir) / "app")}, clear=True):
                    self.assertEqual(resolve_local_langpro_root(), repo_vendor)

    def test_sibling_wins_over_app_vendor(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "repo"
            sibling = project_root.parent / "LangPro"
            app_vendor = Path(tmp_dir) / "app" / "vendor" / "LangPro"
            sibling.mkdir(parents=True)
            app_vendor.mkdir(parents=True)

            with patch("kbprojection.settings.PROJECT_ROOT", project_root):
                with patch.dict(os.environ, {"KBPROJECTION_APP_DIR": str(Path(tmp_dir) / "app")}, clear=True):
                    self.assertEqual(resolve_local_langpro_root(), sibling)

    def test_app_vendor_used_when_higher_priority_paths_absent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "repo"
            app_vendor = Path(tmp_dir) / "app" / "vendor" / "LangPro"
            app_vendor.mkdir(parents=True)

            with patch("kbprojection.settings.PROJECT_ROOT", project_root):
                with patch.dict(os.environ, {"KBPROJECTION_APP_DIR": str(Path(tmp_dir) / "app")}, clear=True):
                    self.assertEqual(resolve_local_langpro_root(), app_vendor)

    def test_missing_error_includes_search_paths_and_auto_clone_guidance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "repo"
            app_dir = Path(tmp_dir) / "app"
            with patch("kbprojection.settings.PROJECT_ROOT", project_root):
                with patch.dict(os.environ, {"KBPROJECTION_APP_DIR": str(app_dir)}, clear=True):
                    message = format_local_langpro_missing_error()

        self.assertIn("Local LangPro checkout not found.", message)
        self.assertIn(str(project_root / "vendor" / "LangPro"), message)
        self.assertIn(str(project_root.parent / "LangPro"), message)
        self.assertIn(str(app_dir / "vendor" / "LangPro"), message)
        self.assertIn("KBPROJECTION_LANGPRO_AUTO_CLONE=1", message)

    @patch("kbprojection.settings.subprocess.run")
    def test_auto_clone_attempts_branch_clone(self, mock_run):
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "repo"
            app_dir = Path(tmp_dir) / "app"
            destination = app_dir / "vendor" / "LangPro"
            env = {
                "KBPROJECTION_APP_DIR": str(app_dir),
                "KBPROJECTION_LANGPRO_AUTO_CLONE": "1",
            }
            with patch("kbprojection.settings.PROJECT_ROOT", project_root):
                with patch.dict(os.environ, env, clear=True):
                    self.assertEqual(resolve_local_langpro_root(allow_clone=True), destination)

        first_call = mock_run.call_args_list[0].args[0]
        self.assertEqual(
            first_call,
            ["git", "clone", "--branch", "nl", DEFAULT_LANGPRO_REPO, str(destination)],
        )

    @patch("kbprojection.settings.subprocess.run")
    def test_auto_clone_falls_back_to_checkout_when_branch_clone_fails(self, mock_run):
        mock_run.side_effect = [
            Mock(returncode=128, stdout="", stderr="branch not found"),
            Mock(returncode=0, stdout="", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "repo"
            app_dir = Path(tmp_dir) / "app"
            env = {
                "KBPROJECTION_APP_DIR": str(app_dir),
                "KBPROJECTION_LANGPRO_AUTO_CLONE": "true",
            }
            with patch("kbprojection.settings.PROJECT_ROOT", project_root):
                with patch.dict(os.environ, env, clear=True):
                    resolve_local_langpro_root(allow_clone=True)

        self.assertEqual(mock_run.call_args_list[1].args[0], ["git", "clone", DEFAULT_LANGPRO_REPO, str(app_dir / "vendor" / "LangPro")])
        self.assertEqual(mock_run.call_args_list[2].args[0], ["git", "checkout", "nl"])


if __name__ == "__main__":
    unittest.main()

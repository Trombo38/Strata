"""#533 multi-GPU: le nom de carte reste le numero nvidia-smi, et le decalage bus PCI
ne doit jamais deplacer la reserve sur la mauvaise carte."""
import unittest
from unittest import mock

from serve.server import Service


def service_with(cvd):
    """Un Service dont le moteur tourne avec CUDA_VISIBLE_DEVICES=cdv (pose avec
    CUDA_DEVICE_ORDER=PCI_BUS_ID par child_env) : index i = ordinal moteur i."""
    svc = Service.__new__(Service)
    eng = mock.Mock()
    eng.spawn = ("exe", [], None, None, {"CUDA_VISIBLE_DEVICES": cvd}, False, None)
    svc.engine = eng
    return svc


class BusSafeMap(unittest.TestCase):
    def test_pci_inverted_pair(self):
        # cette machine : nvidia-smi 0 = 5060 Ti (bus 01), 1 = A5000 (bus 02);
        # la repartition ecrit CUDA_VISIBLE_DEVICES="1,0" -> l'ordinal moteur 0 est la carte 1
        svc = service_with("1,0")
        self.assertEqual(svc.vram_gpu_ordinals(), [1, 0])
        self.assertEqual(svc.vram_ordinal(None), 0)      # carte principale
        self.assertEqual(svc.vram_ordinal(1), 0)         # nvidia-smi 1 (A5000) -> ordinal 0
        self.assertEqual(svc.vram_ordinal(0), 1)         # nvidia-smi 0 (5060 Ti) -> ordinal 1

    def test_plain_order(self):
        svc = service_with("0,1")
        self.assertEqual(svc.vram_ordinal(0), 0)
        self.assertEqual(svc.vram_ordinal(1), 1)

    def test_subset(self):
        # un sous-ensemble de cartes : CUDA_VISIBLE_DEVICES="2" -> l'unique ordinal 0 est la carte 2
        svc = service_with("2")
        self.assertEqual(svc.vram_ordinal(2), 0)

    def test_unnamed_card_is_an_error_not_a_guess(self):
        svc = service_with("1,0")
        with self.assertRaises(ValueError) as x:
            svc.vram_ordinal(7)
        self.assertIn("not a card this engine runs on", str(x.exception))
        self.assertIn("1, 0", str(x.exception))

    def test_no_gpu_list_is_one_card(self):
        eng = mock.Mock()
        eng.spawn = ("exe", [], None, None, {}, False, None)
        svc = Service.__new__(Service)
        svc.engine = eng
        self.assertEqual(svc.vram_gpu_ordinals(), [])
        self.assertEqual(svc.vram_ordinal(None), 0)

    def test_bad_gpu_values(self):
        svc = service_with("1,0")
        for bad in (True, "0", 1.5, -1):
            with self.assertRaises(ValueError):
                svc.vram_ordinal(bad)


class EngineLine(unittest.TestCase):
    def test_gpu_is_on_the_line_only_when_asked(self):
        import io, queue
        from serve.server import StrataEngine
        e = StrataEngine("missing-executable", [], lazy=True)
        e.proc = mock.Mock()
        e.proc.poll.return_value = None
        e.proc.stdin = io.StringIO()
        e.ended = False
        e.lines = queue.Queue()
        e.lines.put("VRAM gpu=1 reserve_mib=6000 expert_slots=1200\n")
        out = e.vram(6000, gpu=1)
        self.assertEqual(e.proc.stdin.getvalue(), "VRAM 1 6000\n")
        self.assertEqual(out["gpu"], 1)
        self.assertEqual(out["expert_slots"], 1200)

    def test_main_card_keeps_the_old_line(self):
        import io, queue
        from serve.server import StrataEngine
        e = StrataEngine("missing-executable", [], lazy=True)
        e.proc = mock.Mock()
        e.proc.poll.return_value = None
        e.proc.stdin = io.StringIO()
        e.ended = False
        e.lines = queue.Queue()
        e.lines.put("VRAM gpu=0 reserve_mib=6000 expert_slots=1200\n")
        e.vram(6000)
        self.assertEqual(e.proc.stdin.getvalue(), "VRAM 6000\n")


class DeviceTable(unittest.TestCase):
    """gpu_device_table: au demarrage, la table ordinal moteur -> numero nvidia-smi -> bus -> nom,
    silencieuse quand il n'y a rien a mapper (mono-carte, pas de CUDA_VISIBLE_DEVICES, pas de nvidia-smi)."""
    def test_split_maps_names_and_buses(self):
        from serve.server import gpu_device_table
        with mock.patch("serve.server.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout=(
                "0, 00000000:01:00.0, NVIDIA GeForce RTX 5060 Ti\n"
                "1, 00000000:02:00.0, NVIDIA RTX A5000\n"))
            table = gpu_device_table({"CUDA_VISIBLE_DEVICES": "1,0"})
        self.assertIn("engine card 0 = nvidia-smi 1 (bus 00000000:02:00.0) NVIDIA RTX A5000", table)
        self.assertIn("engine card 1 = nvidia-smi 0 (bus 00000000:01:00.0) NVIDIA GeForce RTX 5060 Ti", table)

    def test_silent_when_nothing_to_map(self):
        from serve.server import gpu_device_table
        self.assertIsNone(gpu_device_table({}))
        self.assertIsNone(gpu_device_table({"CUDA_VISIBLE_DEVICES": "0"}))

    def test_silent_without_nvidia_smi(self):
        from serve.server import gpu_device_table
        with mock.patch("serve.server.subprocess.run") as run:
            run.side_effect = OSError("no nvidia-smi")
            self.assertIsNone(gpu_device_table({"CUDA_VISIBLE_DEVICES": "1,0"}))


class NonMainCardResize(unittest.TestCase):
    """#533 increment 2 : avec le modele charge, une carte non-main (layer split) se resize ;
    la reserve y va par son ordinal, et la reponse porte le numero nvidia-smi demande."""

    def _loaded_service(self, cvd):
        svc = service_with(cvd)
        svc.loaded = lambda: True
        svc.fifo = mock.Mock()
        svc.fifo.acquire.return_value = True
        svc.batch = 0
        svc.vision = mock.Mock()
        svc.vision.elastic_mib = 2048
        svc.vision_held = False
        svc.vram_reserve = None
        svc.vram_reserve_asked = False
        # le moteur echo gpu=<son ordinal CUDA interne> sur sa ligne VRAM: le mock le reproduit,
        # sinon le test ne voit pas l'ecrasement du numero nvidia-smi par **out
        svc.engine.vram = mock.Mock(return_value={
            "gpu": 1, "vram_free_mib": 6000, "expert_cache_mib": 3000,
            "expert_cache_full_mib": 5000, "expert_slots": 900})
        return svc

    def test_non_main_card_routes_by_ordinal(self):
        # nvidia-smi 0 = 5060 Ti = ordinal moteur 1 (CUDA_VISIBLE_DEVICES="1,0")
        svc = self._loaded_service("1,0")
        out = svc.vram(6000, gpu=0)
        # engine.vram recoit l'ordinal 1, pas le numero nvidia-smi 0
        self.assertEqual(svc.engine.vram.call_args.kwargs.get("gpu"), 1)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["gpu"], 0)          # la reponse renvoie le numero nvidia-smi demande,
                                                 # pas le gpu=0/1 (ordinal) que le moteur echo

    def test_main_card_keeps_the_engine_gpu_from_lying(self):
        # le cas reel de julie : carte principale, le moteur echo gpu=0 (son ordinal),
        # la carte principale est nvidia-smi 1 (CUDA_VISIBLE_DEVICES="1,0") -> reponse gpu: 1
        svc = self._loaded_service("1,0")
        svc.engine.vram = mock.Mock(return_value={
            "gpu": 0, "vram_free_mib": 1728, "expert_cache_mib": 16384,
            "expert_cache_full_mib": 17630, "expert_slots": 11939, "expert_slots_full": 12848})
        out = svc.vram(2000)
        self.assertNotIn("gpu", svc.engine.vram.call_args.kwargs)   # main card: appel 1 argument
        self.assertEqual(out["gpu"], 1)          # nvidia-smi, pas l'ordinal 0 que le moteur echo
        self.assertEqual(out["vram_free_mib"], 1728)

    def test_main_card_uses_the_old_call(self):
        svc = self._loaded_service("1,0")
        out = svc.vram(6000)                     # None = carte principale = ordinal 0
        # main card: engine.vram(want) sans gpu= (moteur d'avant le changement)
        self.assertNotIn("gpu", svc.engine.vram.call_args.kwargs)
        self.assertEqual(out["gpu"], 1)          # nvidia-smi de la carte principale = out[0] = 1

    def test_non_main_card_needs_the_model_loaded(self):
        svc = service_with("1,0")
        svc.loaded = lambda: False
        with self.assertRaises(ValueError) as x:
            svc.vram(6000, gpu=0)
        self.assertIn("only when the model is loaded", str(x.exception))


if __name__ == "__main__":
    unittest.main()

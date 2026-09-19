import unittest
import numpy as np
import torch
from ctc_data import curve_features, minimum_frames
from ctc_train import CTCModel, collapse, edit_distance


class CTCTests(unittest.TestCase):
    def test_ctc_repeated_labels_need_blank(self):
        self.assertEqual(collapse([1, 1, 0, 1, 2, 2, 0]), [1, 1, 2])
        self.assertEqual(minimum_frames([1, 1, 2, 2]), 6)
        self.assertEqual(edit_distance([1, 2], [1, 3, 2]), 1)

    def test_curve_features_preserve_gap_and_time(self):
        strokes = [[[0, 0, 0], [0, 1, 10], [0, 2, 20]],
                   [[2, 2, 30], [2, 3, 40]]]
        x = curve_features(strokes)
        self.assertEqual(x.shape[1], 10)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[:, -1].tolist(), [1, 0, 1])
        self.assertGreater(x[1, 0], 0)
        self.assertGreater(x[0, 6:9].sum(), 0)
        scaled = [[[p[0]*10+40, p[1]*10-20, p[2]] for p in s] for s in strokes]
        np.testing.assert_allclose(x, curve_features(scaled), atol=1e-5)

    def test_model_padding_and_finite_ctc_gradients(self):
        torch.manual_seed(1)
        model = CTCModel(8, dim=32, layers=2, heads=4, ff=64, dropout=0).eval()
        x = torch.randn(2, 12, 10)
        lengths = torch.tensor([12, 8])
        changed = x.clone()
        changed[1, 8:] = 1000
        y = model(x, lengths)
        z = model(changed, lengths)
        self.assertTrue(torch.allclose(y[1, :8], z[1, :8], atol=1e-5))
        target = torch.tensor([1, 1, 2, 3])
        loss = torch.nn.functional.ctc_loss(y.log_softmax(-1).transpose(0, 1),
                    target, lengths, torch.tensor([2, 2]), blank=0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.input.weight.grad).all())


if __name__ == '__main__':
    unittest.main()

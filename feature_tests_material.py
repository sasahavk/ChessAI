import inspect, chess
from feature_extractor_material import MaterialFeatureExtractor

class FeatureTestsMaterial:
    def test_material_balance_1(self):
        board = chess.Board("k7/8/8/8/8/8/8/KPNBRQ2 w - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_material_balance()
        expected = [1, 3, 3, 5, 9]
        print("---test_material_balance_1---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_material_balance_2(self):
        board = chess.Board("kpnq4/8/8/8/8/8/8/KBR5 w - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_material_balance()
        expected = [-1, -3, 3, 5, -9]
        print("---test_material_balance_2---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_mobility_balance_1(self):
        board = chess.Board("8/8/8/8/8/8/8/NPBRQK2 w - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_mobility_balance()
        expected = [2, 2, 7, 7, 14, 4]
        print("---test_mobility_balance_1---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_mobility_balance_2(self):
        board = chess.Board("8/8/8/8/8/8/8/NPBRQK2 b - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_mobility_balance()
        expected = [-2, -2, -7, -7, -14, -4]
        print("---test_mobility_balance_2---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_mobility_safe_balance_1(self):
        board = chess.Board("8/8/8/8/8/8/8/Pn6 w - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_mobility_safe_balance()
        expected = [1, -3, 0, 0, 0, 0]
        print("---test_mobility_safe_balance_1---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_attack_balance_1(self):
        board = chess.Board("8/8/8/8/8/8/8/NPBRQK2 w - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_attack_balance()
        expected = [2, 2, 7, 9, 16, 5]
        print("---test_attack_balance_1---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_threat_balance_1(self):
        #only white pieces on board, so threat balance should be 0
        board = chess.Board("8/8/8/8/8/8/8/NPBRQK2 w - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_threat_balance()
        expected = [0] * 6
        print("---test_threat_balance_1---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_threat_balance_2(self):
        #black's knight and pawn each attacks 2 pieces, white's knight can only attack black's knight
        board = chess.Board("8/8/8/8/8/1n6/1p6/RBNK4 b - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_threat_balance()
        expected = [2, 1, 0, 0, 0, 0]
        print("---test_threat_balance_2---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_defense_balance_1(self):
        #rook protected by queen, queen protected by rook
        board = chess.Board("8/8/8/8/8/8/8/RQ6 w - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_defense_balance()
        expected = [0, 0, 0, 1, 1, 0]
        print("---test_defense_balance_1---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_defense_balance_2(self):
        #rook not protected by knight, knight protected by rook
        board = chess.Board("8/8/8/8/8/8/8/RN6 w - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_defense_balance()
        expected = [0, 1, 0, 0, 0, 0]
        print("---test_defense_balance_2---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_defense_balance_3(self):
        #two rooks protected by one knight, two rooks protected by each other
        board = chess.Board("8/8/8/8/8/1N6/8/R1R5 w - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_defense_balance()
        expected = [0, 0, 0, 4, 0, 0]
        print("---test_defense_balance_3---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_defense_balance_4(self):
        #both black and white have a rook protecting their knights, canceling each other out
        board = chess.Board("nr6/8/8/8/8/8/8/RN6 w - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_defense_balance()
        expected = [0] * 6
        print("---test_defense_balance_4---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def test_defense_balance_5(self):
        #black's knight protected by two rooks, white's knight protected by one rook
        board = chess.Board("rnr5/8/8/8/8/8/8/NR6 w - - 0 1")
        scores = MaterialFeatureExtractor(board, 0).ft_defense_balance()
        expected = [0, -1, 0, 0, 0, 0]
        print("---test_defense_balance_5---")
        print("Expected:", expected)
        if scores == expected: print("PASS")
        else: print("Got:", scores)
        print("")

    def run(self):
        for name, method in sorted(inspect.getmembers(self, predicate=inspect.ismethod)):
            if not name.startswith("test_"): continue
            method()

def main():
    FeatureTestsMaterial().run()

if __name__ == "__main__": main()
# Trabalho de IA - Multilayer Perceptron (MLP)
# Integrantes:
#   - Theo Djrdjrjan Brito - No USP: 13688367
#   - Nome Completo - No USP: 00000000
#
# Metricas para avaliacao de classificacao BINARIA: matriz de confusao
# (VP/VN/FP/FN), acuracia, precisao, revocacao e F1-score. Usadas nos testes
# das portas logicas (AND, OR, XOR).


def matriz_confusao_binaria(reais, previstos, positivo=1):
    # Conta os 4 casos comparando rotulo real x previsto:
    #   VP: real e previsto positivos | VN: real e previsto negativos
    #   FP: previu positivo, era negativo | FN: previu negativo, era positivo
    vp = vn = fp = fn = 0
    for real, prev in zip(reais, previstos):
        if real == positivo and prev == positivo:
            vp += 1
        elif real != positivo and prev != positivo:
            vn += 1
        elif real != positivo and prev == positivo:
            fp += 1
        else:
            fn += 1
    return vp, vn, fp, fn


def avaliar_binario(reais, previstos, positivo=1):
    # Calcula e imprime a matriz de confusao e as metricas da classificacao
    # binaria. As divisoes sao protegidas contra denominador zero.
    vp, vn, fp, fn = matriz_confusao_binaria(reais, previstos, positivo)
    total = vp + vn + fp + fn

    acuracia = (vp + vn) / total if total else 0.0
    precisao = vp / (vp + fp) if (vp + fp) else 0.0
    revocacao = vp / (vp + fn) if (vp + fn) else 0.0
    f1 = (2 * precisao * revocacao / (precisao + revocacao)
          if (precisao + revocacao) else 0.0)

    print(f"  Matriz de confusao (classe positiva = {positivo}):")
    print(f"              Prev 1   Prev 0")
    print(f"     Real 1   {vp:>6}   {fn:>6}")
    print(f"     Real 0   {fp:>6}   {vn:>6}")
    print(f"  Acuracia:  {acuracia:.3f}")
    print(f"  Precisao:  {precisao:.3f}")
    print(f"  Revocacao: {revocacao:.3f}")
    print(f"  F1-score:  {f1:.3f}")

    return acuracia, precisao, revocacao, f1

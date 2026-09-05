/**
 * Похожесть двух строк по алгоритму difflib.SequenceMatcher.ratio из Python.
 *
 * Нужна ровно для одного: отсеивать подсказку, которая почти повторяет недавнюю.
 * Взять «похожий» алгоритм нельзя — порог 0.85 подбирался под этот, и с другой
 * мерой он означал бы другое. Совпадение с оригиналом проверено эталоном
 * (similarity.test.ts).
 *
 * Считается 2·M / (|a| + |b|), где M — сумма длин совпадающих блоков, найденных
 * рекурсивным поиском самого длинного общего куска.
 */

/**
 * Позиции символов b. При длине от 200 символов difflib объявляет «мусорными»
 * те, что встречаются чаще чем в 1 % позиций, и исключает их из поиска — иначе
 * на длинных строках пробелы съедают всё время. Поведение воспроизводим: без
 * него длинные подсказки сравнивались бы иначе, чем на сервере.
 */
function индексы(b: string): { b2j: Map<string, number[]>; мусор: Set<string> } {
  const b2j = new Map<string, number[]>();
  for (let i = 0; i < b.length; i += 1) {
    const список = b2j.get(b[i]);
    if (список) список.push(i);
    else b2j.set(b[i], [i]);
  }
  const мусор = new Set<string>();
  if (b.length >= 200) {
    const потолок = Math.trunc(b.length / 100) + 1;
    for (const [символ, места] of b2j) {
      if (места.length > потолок) {
        мусор.add(символ);
        b2j.set(символ, []);
      }
    }
  }
  return { b2j, мусор };
}

interface Блок {
  a: number;
  b: number;
  size: number;
}

/** Самый длинный общий кусок в a[alo:ahi] и b[blo:bhi]. */
function самыйДлинный(
  a: string, b: string, alo: number, ahi: number, blo: number, bhi: number,
  b2j: Map<string, number[]>, мусор: Set<string>,
): Блок {
  let besti = alo;
  let bestj = blo;
  let bestsize = 0;
  let j2len = new Map<number, number>();

  for (let i = alo; i < ahi; i += 1) {
    const newj2len = new Map<number, number>();
    for (const j of b2j.get(a[i]) ?? []) {
      if (j < blo) continue;
      if (j >= bhi) break;
      const k = (j2len.get(j - 1) ?? 0) + 1;
      newj2len.set(j, k);
      if (k > bestsize) {
        besti = i - k + 1;
        bestj = j - k + 1;
        bestsize = k;
      }
    }
    j2len = newj2len;
  }

  // Частые символы исключены из поиска, поэтому найденный кусок надо расширить
  // по краям — сначала обычными символами, потом отсеянными. Без этого длинные
  // строки сравниваются иначе, чем в difflib: там кусок дотягивается до
  // настоящей границы совпадения.
  const дотянуть = (частые: boolean): void => {
    while (
      besti > alo && bestj > blo &&
      мусор.has(b[bestj - 1]) === частые && a[besti - 1] === b[bestj - 1]
    ) {
      besti -= 1;
      bestj -= 1;
      bestsize += 1;
    }
    while (
      besti + bestsize < ahi && bestj + bestsize < bhi &&
      мусор.has(b[bestj + bestsize]) === частые &&
      a[besti + bestsize] === b[bestj + bestsize]
    ) {
      bestsize += 1;
    }
  };
  дотянуть(false);
  дотянуть(true);

  return { a: besti, b: bestj, size: bestsize };
}

function блоки(a: string, b: string, b2j: Map<string, number[]>, мусор: Set<string>): Блок[] {
  const очередь: [number, number, number, number][] = [[0, a.length, 0, b.length]];
  const найденные: Блок[] = [];
  while (очередь.length > 0) {
    const [alo, ahi, blo, bhi] = очередь.pop()!;
    const блок = самыйДлинный(a, b, alo, ahi, blo, bhi, b2j, мусор);
    if (блок.size === 0) continue;
    найденные.push(блок);
    if (alo < блок.a && blo < блок.b) очередь.push([alo, блок.a, blo, блок.b]);
    if (блок.a + блок.size < ahi && блок.b + блок.size < bhi) {
      очередь.push([блок.a + блок.size, ahi, блок.b + блок.size, bhi]);
    }
  }
  return найденные;
}

/** Похожесть от 0 до 1. Пустые строки считаются одинаковыми — как в Python. */
export function similarityRatio(a: string, b: string): number {
  if (a.length === 0 && b.length === 0) return 1;
  const { b2j, мусор } = индексы(b);
  const совпало = блоки(a, b, b2j, мусор).reduce((сумма, блок) => сумма + блок.size, 0);
  return (2 * совпало) / (a.length + b.length);
}

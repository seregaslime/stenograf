// Линтер клиента. Набор правил выбран узко, как и у сервера (server/ruff.toml).
//
// Задача — ловить ошибки, которых не видит TypeScript: он проверяет типы, но не
// замечает условно вызванный хук или забытую зависимость эффекта. Всё, что про
// вкус и оформление, намеренно выключено: спорить о переносах в ревью дороже,
// чем польза от единообразия.
//
// Расширение .mjs, а не .js: конфиг написан модулями, а в package.json проекта
// "type" не задан — иначе node ругается при каждом запуске.
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**", "release/**", "node_modules/**", "public/**"] },
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: { "react-hooks": reactHooks },
    rules: {
      // Главное, ради чего всё затевалось: хук, вызванный по условию, ломает
      // React молча и невоспроизводимо.
      "react-hooks/rules-of-hooks": "error",
      // exhaustive-deps оставлен предупреждением: в LivePage несколько эффектов
      // с пустым списком зависимостей стоят намеренно (подписка один раз за
      // жизнь компонента), и правило на них ругается по делу своего замысла.
      "react-hooks/exhaustive-deps": "warn",
      // Подчёркивание перед именем — «аргумент нужен для сигнатуры, но не телу».
      // Так объявлены моки fetch в тестах: без параметров у мока пустой тип
      // аргументов, и обращение к mock.calls[0][1] не проходит проверку типов.
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
    },
  },
);

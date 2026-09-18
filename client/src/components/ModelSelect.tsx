/**
 * Выбор модели из списка скачанных в Ollama.
 *
 * Список приходит от самой Ollama, и его может не быть: адрес не отвечает, её
 * не запустили, моделей не скачали. Тогда вместо пустого списка, из которого
 * ничего не выбрать, показывается поле ввода — иначе человек с недоступной
 * Ollama не смог бы даже сохранить имя модели, которое он знает.
 *
 * Выбранная модель остаётся в списке, даже если её нет среди скачанных: она
 * могла уехать вместе с Ollama на другую машину, и молча подменять выбор
 * человека на первую попавшуюся нельзя.
 */
export default function ModelSelect({
  value,
  models,
  placeholder,
  onChange,
}: {
  value: string;
  models: string[];
  placeholder: string;
  onChange: (value: string) => void;
}) {
  if (models.length === 0) {
    return (
      <input
        className="input"
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  }
  return (
    <select className="input" value={value} onChange={(event) => onChange(event.target.value)}>
      {value && !models.includes(value) && <option value={value}>{value} (не скачана)</option>}
      {!value && <option value="">— выберите модель —</option>}
      {models.map((model) => (
        <option key={model} value={model}>
          {model}
        </option>
      ))}
    </select>
  );
}

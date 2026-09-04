/** Which language a piece of text is in, for the two this app writes.
 *
 *  One definition, used twice, and that is the point. The server decides the
 *  language of the ANSWER from the language of the QUESTION; the panel then
 *  writes its own sentences around that answer -- the note about folded
 *  handles, and anything added later. Two copies of this rule drift, and the
 *  drift shows up as an English sentence sitting under an Indonesian reply,
 *  which is exactly the mixing the language rule exists to prevent. Measured
 *  on 24 August 2026, in a reply about 60 mismatched plots.
 */

/** Words that only appear in Indonesian, common enough to catch a short
 *  question. Function words rather than nouns: a question can be about a
 *  mosque in either language. */
export const INDONESIAN_MARKERS =
  /\b(yang|dan|dari|untuk|dengan|adalah|tidak|ada|ini|itu|saya|anda|apa|apakah|berapa|dimana|bagaimana|tolong|mohon|kenapa|mengapa|atau|juga|bisa|sudah|belum|akan|pada|nya)\b/i;

export type Language = "id" | "en";

/** The language of a piece of text.
 *
 *  Applied to a QUESTION on the server and to an ANSWER in the panel. Both
 *  work, and for the same reason: the markers are function words, so a
 *  sentence of any length in either language carries several of them.
 */
export function languageOf(text: string): Language {
  return INDONESIAN_MARKERS.test(text) ? "id" : "en";
}

/** The note shown when more handles were mentioned than can become buttons.
 *
 *  Nothing is dropped -- the handles are still in the text and every one of
 *  those objects is marked in the drawing. The note says so, in the language
 *  the answer is written in.
 */
export function foldedHandlesNote(count: number, language: Language): string {
  if (language === "id") {
    return (
      `${count} handle lain di atas ditulis sebagai teks biasa, bukan tombol; ` +
      `semuanya tetap tertandai di gambar.`
    );
  }
  return (
    `${count} further handle${count === 1 ? "" : "s"} above ` +
    `are plain text rather than buttons; every one of them is marked in ` +
    `the drawing.`
  );
}

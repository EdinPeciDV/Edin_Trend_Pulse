/**
 * src/components/Terms.jsx
 * -------------------------------------------------------------------
 * Terms & Conditions. This is the single place where liability,
 * risk-disclosure, and "not investment advice" language lives — other
 * screens should link here instead of restating it.
 *
 * Legal review recommended before relying on this in a dispute. The
 * jurisdiction/governing-law and contact fields below are placeholders
 * — fill them in with your actual details.
 * -------------------------------------------------------------------
 */

const LAST_UPDATED = '12 August 2026';
const GOVERNING_LAW = 'Kosovo'; // TODO: set to your actual jurisdiction
const CONTACT_EMAIL = 'support@trendpulse.app'; // TODO: set to your actual contact address

function Section({ title, children }) {
  return (
    <section className="panel px-4 py-4">
      <h2 className="label-amber">{title}</h2>
      <div className="mt-2 max-w-prose space-y-2 text-tick normal-case leading-relaxed text-ink-muted">
        {children}
      </div>
    </section>
  );
}

export default function Terms() {
  return (
    <div className="mx-auto max-w-3xl space-y-4 pb-10">
      <div>
        <h1 className="font-sans text-xl font-semibold text-ink">
          Terms &amp; Conditions
        </h1>
        <p className="mt-1 text-micro uppercase text-ink-faint">
          Last updated {LAST_UPDATED}
        </p>
      </div>

      <Section title="1. Not financial advice">
        <p>
          TrendPulse computes standard technical indicators (RSI, SMA,
          Bollinger Bands, VWAP) against recent market data and displays where
          price sits relative to them, together with a heuristic confidence
          score. That is the entire product.
        </p>
        <p>
          Nothing on this site or produced by this service constitutes
          financial, investment, trading, tax, or legal advice, and nothing
          here is a recommendation or solicitation to buy, sell, or hold any
          asset. TrendPulse is not a broker-dealer, investment adviser,
          financial planner, or fiduciary of any kind, is not registered as
          such with any regulator, and does not hold, execute, or route any
          trades or funds on your behalf. Every decision you make using this
          site is made entirely by you, at your own discretion.
        </p>
      </Section>

      <Section title="2. No accuracy or performance guarantee">
        <p>
          Indicator values, confidence scores, and any prediction log
          statistics are generated algorithmically from third-party market
          data (currently Binance for crypto, Twelve Data for forex) and are
          provided for informational and educational purposes only. Market
          data feeds can be delayed, incomplete, or wrong, and the
          calculations built on top of them inherit those errors.
        </p>
        <p>
          Past performance shown anywhere on this site — including any
          historical hit rate or prediction log — is not indicative of
          future results and is not a guarantee, projection, or assurance of
          any future outcome. Confidence percentages measure agreement
          between the tool's own internal rules; they are not a probability
          of any market outcome and have not been independently audited or
          validated against real trading returns net of fees, spreads,
          slippage, or funding costs.
        </p>
      </Section>

      <Section title="3. Assumption of risk">
        <p>
          Trading cryptocurrency and leveraged forex carries a substantial
          risk of loss and is not suitable for every investor. You could
          lose some, all, or more than your original investment. By using
          TrendPulse, you acknowledge that you understand these risks and
          that you are solely responsible for evaluating the merits and
          risks of any decision before acting on it, including consulting a
          licensed, qualified professional where appropriate.
        </p>
      </Section>

      <Section title="4. Service provided &quot;as is&quot;">
        <p>
          The service is provided on an "as is" and "as available" basis,
          without warranties of any kind, whether express, implied, or
          statutory, including without limitation any implied warranties of
          merchantability, fitness for a particular purpose, title,
          non-infringement, accuracy, or uninterrupted or error-free
          operation. We do not warrant that the service will be
          uninterrupted, timely, secure, or free of errors, that data feeds
          or third-party integrations will remain available, or that any
          defect will be corrected.
        </p>
      </Section>

      <Section title="5. Limitation of liability">
        <p>
          To the maximum extent permitted by applicable law, in no event
          will TrendPulse, its operator, or its affiliates, officers,
          employees, or agents be liable for any direct, indirect,
          incidental, special, consequential, exemplary, or punitive
          damages — including without limitation lost profits, lost trading
          gains, lost data, or trading or investment losses of any kind —
          arising out of or in connection with your access to, use of, or
          reliance on the service, its content, or any output it produces,
          even if advised of the possibility of such damages, and regardless
          of the legal theory on which the claim is based.
        </p>
        <p>
          Where applicable law does not allow the exclusion or limitation of
          certain damages, the above limitation applies to the fullest
          extent permitted, and our total aggregate liability for any claim
          arising from your use of the service will not exceed the greater
          of (a) the amount you paid us, if any, for the service in the
          twelve months preceding the claim, or (b) fifty US dollars
          (US$50).
        </p>
      </Section>

      <Section title="6. Indemnification">
        <p>
          You agree to indemnify, defend, and hold harmless TrendPulse, its
          operator, and its affiliates, officers, employees, and agents from
          and against any claims, liabilities, damages, losses, and
          expenses, including reasonable legal fees, arising out of or in
          any way connected with your access to or use of the service, your
          trading or investment decisions, or your violation of these
          Terms.
        </p>
      </Section>

      <Section title="7. Subscriptions and billing">
        <p>
          Paid plans are billed through Paddle, our payments provider and
          merchant of record, which handles checkout, invoicing, tax, and
          payment processing. Your use of paid features is also subject to
          Paddle's own terms and privacy policy for the payment transaction
          itself. Plan limits, features, and pricing may change; we will
          make reasonable efforts to communicate material changes in
          advance. Cancellations, refunds, and billing disputes are handled
          per the checkout provider's stated policy at the time of
          purchase.
        </p>
      </Section>

      <Section title="8. Accounts and eligibility">
        <p>
          You must be at least 18 years old, or the age of legal majority in
          your jurisdiction, and legally permitted to access financial
          market data and, where relevant, trade the relevant instruments in
          your jurisdiction, to use this service. You are responsible for
          keeping your account credentials confidential and for all
          activity under your account.
        </p>
      </Section>

      <Section title="9. Changes to the service and these Terms">
        <p>
          We may modify, suspend, or discontinue any part of the service at
          any time, with or without notice. We may update these Terms from
          time to time; the "Last updated" date above reflects the most
          recent revision. Continued use of the service after a change
          takes effect constitutes acceptance of the revised Terms.
        </p>
      </Section>

      <Section title="10. Governing law">
        <p>
          These Terms are governed by the laws of {GOVERNING_LAW}, without
          regard to its conflict-of-laws principles, and any dispute arising
          from these Terms or the service will be subject to the exclusive
          jurisdiction of the courts of that jurisdiction, except where
          applicable consumer-protection law requires otherwise.
        </p>
      </Section>

      <Section title="11. Contact">
        <p>
          Questions about these Terms can be sent to{' '}
          <a
            href={`mailto:${CONTACT_EMAIL}`}
            className="text-amber hover:underline"
          >
            {CONTACT_EMAIL}
          </a>
          .
        </p>
      </Section>
    </div>
  );
}
